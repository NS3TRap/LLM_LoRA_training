from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os

parser = argparse.ArgumentParser(description='Chatting with fine-tune causal LM with LoRA (PEFT)')
parser.add_argument('--base_model', type=str, required=True, help='Название модели(из HF)')
parser.add_argument('--model_dir', type=str, required=True, help='Директория с моделью')
parser.add_argument('--test_file', type=str, required=True, help='Файл с тестовым датасетом')

args = parser.parse_args()

def read_prompts(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        prompts = [line.strip() for line in lines if line.strip()]
    return prompts

def generate_answer(model, tokenizer, model_type, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(f"{model_type}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype="auto",
    device_map="auto",
)

lora_model = PeftModel.from_pretrained(model, args.model_dir)

prompts = read_prompts(os.path.join(args.test_file))

for prompt in prompts:
    generate_answer(model, tokenizer, "Предобученная модель", prompt)
    generate_answer(lora_model, tokenizer, "Дообученная модель", prompt)