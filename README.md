**Перед использование данного скрипта:**

>     1. python -m venv llm_env
>     2. source llm_env/bin/activate
>     3. pip install -r requirements.txt

После необходимо скачать и перенести в директорию со скриптом датасет, на котором будет обучаться LLM. Например:

 - https://huggingface.co/datasets/SiberiaSoft/SiberianPersonaChat
 - https://www.kaggle.com/datasets/grafstor/19-000-russian-poems/data
 - или свой

Также стоит отметить, что main.py парсит в режиме Prompt-Responce! Если необходимо, чтобы нейронка просто обучалась на каком-то тексте, то перепишите часть кода в main.py.
Далее есть 3 команды(train_model.py - обучение модели, chatting.py - общение с моделью в режиме диалога, test_model.py - запуск тестирования с выводом результата до- и предобученной моделями):

>  - python train_model.py --model_name Qwen/Qwen2.5-1.5B --data_path data.jsonl --output_dir ./my-lora-model --use_lora
>  - python chatting.py --model_name Qwen/Qwen2.5-1.5B --model_dir my-lora-model
>  - python test_model.py --model_name Qwen/Qwen2.5-1.5B --model_dir my-lora-model --test_file test_questions.txt

Если установлен make, то можно запускать через Makefile: **make *command name***(например make chat)

Параметры для запуска обучения:

 1. model_name --- это название модели с hugging face. Указывайте те модели, которые не требуют авторизации, иначе будет ошибка(str, default='gpt2').
 2. data_path --- это путь к файлу с данными для обучения. Тут используется jsonl/json, но если будет какой-нибудь другой формат, то можете поменять код загрузки датасета(str, required=True). 
 3. output_dir --- это путь куда сохранится модель после обучения(str, default='./fine-tuned-lora').
 4. per_device_train_batch_size --- размер батча на одно устройство (GPU/CPU)(int, default=4).
 5. num_train_epochs --- количество проходов по датасету(int, default=3).
 6. learning_rate --- начальная скорость обучения(float, default=2e-4).
 7. max_length --- максимальное количество токенов (input+output) после токенизации(int, default=512).
 8. use_lora --- флаг, включающий обучение через LoRA(flag, action='store_true').
 9. lora_r --- размер low-rank матрицы(int, default=8).
 10. lora_alpha --- влияет на вклад LoRA в итоговое изменение весов(int, default=32).
 11. lora_dropout --- dropout, применяемый в LoRA слоях(float, default=0.1).
 12. seed --- сид для детерминированных результатов(int, default=42).

Параметры для тестирования и общения:
1. base_model --- это название модели с hugging face. Указывайте те модели, которые не требуют авторизации, иначе будет ошибка(str, default='gpt2').
2. model_dir --- это путь к папке с дообученой моделью (!ОПЦИОНАЛЬНО ДЛЯ chatting.py!)
3. test_file --- это путь к файлу с промптами для тестирования(txt). Промпты записываются построчно. (!ТОЛЬКО ДЛЯ test_model.py!)

PS. Для общения(chatting.py) необязательно указывать model_dir, тогда LLM будет работать в режиме "как есть". Если указать model_dir(LoRA), то будет использоваться LoRA при обычном общении с LLM.
Путь, который указывает куда именно должна сохраниться модель, должен быть указан в test.py.   

*Если возникнут какие-либо ошибки можете написать в личку, обсудим и исправим репозиторий.*
