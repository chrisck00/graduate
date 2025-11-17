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
task_list = ["arc_challenge", "rte", "piqa"]
# task_list = ["hellaswag", "coqa", "rte", "boolq", "qqp"]

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
# model_path = "./merged_models/merged_lora_arc_challenge"
model_path = "./merged_models/merged_model"
# model_path = "./merged_models/merged_model_arc_open_piqa"

# 평가 실행
results = eval_zero_shot(model_path, task_list, num_shot)
