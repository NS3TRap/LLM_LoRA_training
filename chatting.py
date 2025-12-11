from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os

parser = argparse.ArgumentParser(description='Chatting with fine-tune causal LM with LoRA (PEFT)')
parser.add_argument('--base_model', type=str, required=True, help='Название модели(из HF)')
parser.add_argument('--model_dir', type=str, default='./fine-tuned-lora', help='Директория, с моделью')

args = parser.parse_args()

def chat(model, tokenizer):
    while True:
        prompt = input("Введите вопрос: ")
        if prompt == 'q':
            return

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        print(f"Ответ: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

model_dir = os.path.join(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)

model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype="auto",
    device_map="auto",
)
if args.model_dir != '':
    model = PeftModel.from_pretrained(model, args.model_dir)
    chat(model, tokenizer)
else:
    chat(model, tokenizer)
