import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"  # 경고 제거
os.environ["USE_BITSANDBYTES"] = "0"       # bitsandbytes 사용 금지

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

dataset = "piqa"  # 사용할 데이터셋 이름

base_model_name = "meta-llama/Llama-2-7b-hf"
lora_model_name = f"./lora/{dataset}"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cpu",
    torch_dtype="float16"
)

lora_model = PeftModel.from_pretrained(
    base_model,
    lora_model_name,
    device_map="cpu",
    torch_dtype="float16",
    load_in_8bit=False
)

# Merge & Unload
merged_model = lora_model.merge_and_unload()

# 저장 경로
merged_path = f"./merged_models/merged_lora_{dataset}"
merged_model.save_pretrained(merged_path)

# 토크나이저 저장
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_path)

# 필요 없는 adapter_config.json 삭제
adapter_config_path = os.path.join(merged_path, "adapter_config.json")
if os.path.exists(adapter_config_path):
    os.remove(adapter_config_path)
    print("🗑️ adapter_config.json 삭제 완료")

print("✅ Merge 및 정리 완료!")