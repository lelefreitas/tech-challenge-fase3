# Cole em uma célula do Colab (Runtime > Change runtime type > T4 GPU)
!pip -q install transformers peft trl bitsandbytes accelerate datasets

import torch
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto",
)
msgs = [{"role": "user", "content": "Quais são os sinais de alerta de sepse?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=80)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
