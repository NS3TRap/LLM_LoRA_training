import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel

# -------------------- Утилиты --------------------

def read_json_or_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if not text:
            return []
        # Попытка загрузить как JSON целиком (список)
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        # Иначе обрабатываем как JSONL
        items = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items


def get_default_lora_targets(model_name: str) -> List[str]:
    mn = model_name.lower()
    if 'llama' in mn or 'alpaca' in mn:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if 'gpt' in mn or 'dialo' in mn or 'gpt2' in mn:
        # GPT2-подобные модели: attention и mlp
        return ["c_attn", "c_proj", "c_fc", "c_ffn", "mlp"]
    # По умолчанию — попробовать стандартные
    return ["c_attn", "c_proj"]


# -------------------- Подготовка данных --------------------

def prepare_dataset(items: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 512):
    records = []

    sep = tokenizer.eos_token or tokenizer.sep_token or ""

    for it in items:
        prompt = it.get('prompt', '')
        response = it.get('response', '')
        if prompt is None: prompt = ''
        if response is None: response = ''

        # Собираем последовательность: prompt + sep + response
        full = prompt + (sep if sep else "\n") + response

        enc = tokenizer(full, truncation=True, max_length=max_length)
        input_ids = enc['input_ids']
        attention_mask = enc.get('attention_mask', [1] * len(input_ids))

        # Токенизируем prompt отдельно, чтобы знать длину
        enc_prompt = tokenizer(prompt + (sep if sep else "\n"), truncation=True, max_length=max_length)
        prompt_len = len(enc_prompt['input_ids'])

        # Формируем labels: -100 для prompt части, реальные id для response
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        # Если длины не совпадают — выравниваем
        if len(labels) != len(input_ids):
            # Обрезаем или дополняем labels
            labels = labels[:len(input_ids)] + [-100] * max(0, len(input_ids) - len(labels))

        records.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        })

    ds = Dataset.from_list(records)
    return ds


@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:

        # 1. Собираем списки
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # 2. Находим max длину внутри батча
        max_len = max(x.size(0) for x in input_ids)

        # 3. Паддим вручную
        def pad(tensor, pad_value):
            return torch.nn.functional.pad(
                tensor,
                (0, max_len - tensor.size(0)),
                value=pad_value
            )

        input_ids = torch.stack([pad(x, self.tokenizer.pad_token_id) for x in input_ids])
        attention_mask = torch.stack([pad(x, 0) for x in attention_mask])
        labels = torch.stack([pad(x, -100) for x in labels])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    parser = argparse.ArgumentParser(description='Fine-tune causal LM with LoRA (PEFT)')

    parser.add_argument('--model_name', type=str, default='gpt2', help='Название модели в HF (пример: gpt2, microsoft/DialoGPT-medium, facebook/llama-7b и т.д.)')
    parser.add_argument('--data_path', type=str, required=True, help='Путь к data.json или data.jsonl')
    parser.add_argument('--output_dir', type=str, default='./fine-tuned-lora', help='Куда сохранить модель')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Загрузка токенизатора и модели: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Гарантируем наличие pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем модель
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Настройка LoRA
    if args.use_lora:
        target_modules = get_default_lora_targets(args.model_name)
        print(f"Используем target_modules для LoRA: {target_modules}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Читаем данные
    items = read_json_or_jsonl(args.data_path)
    if not items:
        raise SystemExit("Данные пусты или не удалось прочитать файл")

    # Подготовка датасета
    ds = prepare_dataset(items, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_total_limit=3,
        fp16=False,  # на CPU это недоступно
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    print("Начинаем обучение...")
    trainer.train()

    print("Сохраняем модель...")
    # Сохраняем основную модель и токенизатор
    os.makedirs(args.output_dir, exist_ok=True)
    # Если модель PEFT (LoRA), то используем save_pretrained
    try:
        # PeftModel имеет save_pretrained
        model.save_pretrained(args.output_dir)
    except Exception:
        # Стандартный save
        trainer.save_model(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)

    print(f"Готово. Модель сохранена в {args.output_dir}")


if __name__ == '__main__':
    main()
