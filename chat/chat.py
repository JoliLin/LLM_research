import torch
from transformers import pipeline

model_name = "MediaTek-Research/Breeze-7B-Instruct-v0.1"
#model_name = "HuggingFaceH4/zephyr-7b-beta"

pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
            {"role": "user", "content": "Please describe machine learning briefly."},]

messages = [{"role": "system", "content": "請找出以下字句的關鍵字",},
            {"role": "user", "content": "【RICO baby】MEENE-121度純淨面膜-葡萄柚透淨亮白"}]


prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
