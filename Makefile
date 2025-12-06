train:
	python train_model.py --model_name Qwen/Qwen2.5-1.5B --data_path data.jsonl --output_dir ./my-lora-model --use_lora

chat:
	python chatting.py --base_model Qwen/Qwen2.5-1.5B --model_dir my-lora-model

test:
	python test_model.py --base_model Qwen/Qwen2.5-1.5B --model_dir my-lora-model --test_file test_questions.txt
