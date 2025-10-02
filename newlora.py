import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from evallora import eval_zero_shot
from lm_eval.models.huggingface import HFLM

# 예약된 메모리 중 사용하지 않는 공간 해제
torch.cuda.empty_cache()

# 설정
accelerate = True
# task_list = ["rte","boolq","hellaswag","arc_challenge", "openbookqa","piqa","coqa"]
num_shot = 0
task_list = ["hellaswag", "coqa", "rte"]
# task_list = ["arc_challenge", "openbookqa", "piqa"]

# # 어댑터 경로들
adapter_paths = [
    "./lora/hellaswag/adapter_model.safetensors",
    "./lora/rte/adapter_model.safetensors",
    "./lora/bool_q/adapter_model.safetensors",
    "./lora/arc_challenge/adapter_model.safetensors",
    "./lora/openbookqa/adapter_model.safetensors",
    "./lora/piqa/adapter_model.safetensors",
    "./lora/coqa/adapter_model.safetensors"
]
# model_path = "./merged_models/merged_lora_piqa"
model_path = "./merged_models/merged_model_hella_rte_coqa2"
# model_path = "./merged_models/merged_model_arc_open_piqa"

"""
# base model 경로
base_model_path = "meta-llama/Llama-2-7b-hf"

# 어댑터 병합
merged_adapters = {}
for path in adapter_paths:
    adapter = load_file(path)
    for key, weight in adapter.items():
        if key in merged_adapters:
            if merged_adapters[key].shape != weight.shape:
                raise ValueError(f"Shape mismatch at {key}: {merged_adapters[key].shape} vs {weight.shape}")
            merged_adapters[key] += weight
        else:
            merged_adapters[key] = weight.clone()

# base 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir="/workspace/cache",
    device_map="auto"
)

# 병합 적용
model_sd = model.state_dict()
for key, adapter_weight in merged_adapters.items():
    if key in model_sd:
        if model_sd[key].shape != adapter_weight.shape:
            print(f"⚠️ Shape mismatch: {key} ({model_sd[key].shape} vs {adapter_weight.shape}) → skip")
            continue
        model_sd[key] += adapter_weight.to(model_sd[key].device)
        print(f"✅ Merged: {key}")
    else:
        print(f"❌ Not found in base model: {key}")

model.load_state_dict(model_sd)
merged_path = "./llm_weights"
model.save_pretrained(merged_path)
"""
# 평가 실행
results = eval_zero_shot(model_path, task_list, num_shot)