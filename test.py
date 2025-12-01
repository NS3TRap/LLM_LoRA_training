import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "./my-lora-model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

prompts = [
    "Искусственный интеллект это",
    "Нейронные сети работают потому что",
    "Градиентный спуск используется для того чтобы",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("\nPROMPT:", prompt)
    print("RESULT:", tokenizer.decode(outputs[0], skip_special_tokens=True))
