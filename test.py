from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(base_model)

def read_prompts(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        prompts = [line.strip() for line in lines]
    return prompts

prompts = read_prompts('test_questions.txt')

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

