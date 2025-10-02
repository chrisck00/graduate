import shutil
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from collections import defaultdict

BASE_MODEL = "./base_model"
HELLA_MODEL = "./merged_models/merged_lora_hellaswag"
RTE_MODEL = "./merged_models/merged_lora_rte"
COQA_MODEL = "./merged_models/merged_lora_coqa"
MNLI_MODEL = "./merged_models/merged_lora_mnli_mismatched"
QNLI_MODEL = "./merged_models/merged_lora_qnli"
ARCC_MODEL = "./merged_models/merged_lora_arc_challenge"
OPEN_MODEL = "./merged_models/merged_lora_openbookqa"
PIQA_MODEL = "./merged_models/merged_lora_piqa"
MERGED_MODEL = "./merged_models/merged_model_hella_rte_coqa"

base_model_name = BASE_MODEL
finetuned_model_name1 = HELLA_MODEL
finetuned_model_name2 = RTE_MODEL
finetuned_model_name3 = COQA_MODEL
save_path = MERGED_MODEL

task_models = {
    "lm": BASE_MODEL
}

task_heads = {
    "lm": 1
}

#region Hugging Face 토큰
token = "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"
#endregion

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_role(key: str):
    if "q_proj.weight" in key: return "q_weight"
    #if "k_proj.weight" in key: return "k_weight"
    if "v_proj.weight" in key: return "v_weight"
    #if "o_proj.weight" in key: return "o_weight"
    #if "gate_proj.weight" in key: return "gate_weight"
    #if "up_proj.weight" in key: return "up_weight"
    #if "down_proj.weight" in key: return "down_weight"

    return None  # Ignore others

def compute_role_rms(base_model_name, finetuned_model_name, device="cuda"):
    """
    base_model_name: str, ex) "./models/t5-base"
    finetuned_model_name: str, ex) "./models/t5-base-finetuned-GLUE-MNLI"
    device: "cuda" or "cpu"

    return:
        role_rms: {role: {key: rms}}
        role_minmax: {role: (min_rms, max_rms, mean_rms)}
        delta: {key: delta_tensor (CPU)}
    """
    def load_model(model_name, device):
        print(f"Loading {model_name} to {device}...")
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,             # auto 말고 None
            low_cpu_mem_usage=False
        ).to("cpu")  # 강제로 CPU로 올리기
        return model

    def layer_rms(tensor):
        return 1000*tensor.pow(2).mean().sqrt().item()  # RMS

    # 결과 저장용
    role_rms = defaultdict(dict)
    role_minmax = {}
    delta = {}

    # 1) base 모델 불러와서 CPU에 로드
    base_model = load_model(base_model_name, device)
    base_state = {k: v.detach() for k, v in base_model.state_dict().items()}

    # 2) fine-tuned 모델 불러오기 (CPU)
    ft_model = load_model(finetuned_model_name, device)
    ft_state = {k: v.detach() for k, v in ft_model.state_dict().items()}

    # 3) delta 계산 + 끈 (CPU 저장)
    for k in base_state.keys():
        role = get_role(k)
        if role is not None:
            delta_tensor = ft_state[k] - base_state[k]
            delta[k] = delta_tensor.detach().cpu()  # 최종 저장은 CPU
            rms = layer_rms(delta_tensor)
            role_rms[role][k] = rms

    # 4) role별 min/max/mean
    for role, rms in role_rms.items():
        values = list(rms.values())
        role_minmax[role] = (min(values), max(values), sum(values) / len(values))

    # 메모리 해제
    del base_model, ft_model, base_state, ft_state
    torch.cuda.empty_cache()

    return role_rms, role_minmax, delta

def compute_role_drop_rates(delta: dict, role_rms: dict, combined_minmax: dict, default_drop_rate=0, drop_scale=0):
    """
    delta: {key: tensor}
    role_rms: {role: {key: rms}}
    combined_minmax: {role: (rmin, rmax, rmean)}
    drop_rate, drop_scale

    role_drop_rates = {role: {key: drop_rate}}
    """

    role_drop_rates = defaultdict(dict)

    for k, _ in delta.items():
        role = get_role(k)
        if role is None or role not in combined_minmax:
            continue

        rms_val = role_rms[role][k]
        _, _, rmean = combined_minmax[role]

        drop_rate =  default_drop_rate - (rms_val - rmean) * drop_scale
        role_drop_rates[role][k] = drop_rate

    return role_drop_rates

def apply_delta_drop(delta: dict, role_drop_rates: dict, seed: int = 42):
    """
    delta: {key: tensor}
    role_drop_rates: {role: {key: drop_rate}}

    delta_dropped: {key: tensor}
    """
    delta_dropped = {}

    # seed 고정
    torch.manual_seed(seed)

    for k, delta_tensor in delta.items():
        role = get_role(k)
        if role is None:
            continue

        # key에 해당하는 drop_rate 가져오기
        drop_rate = role_drop_rates[role][k]  # key 단위로 drop_rate이 저장되어 있어야 함

        # GPU에서 처리
        delta_tensor = delta_tensor.to(device)

        # mask 생성 및 drop 적용
        mask = (torch.rand_like(delta_tensor) > drop_rate).float()
        dropped = delta_tensor * mask

        # 결과를 CPU로 내림
        delta_dropped[k] = dropped.cpu()

        # 필요 없으면 원본 delta_tensor는 CPU로 내리거나 삭제 가능
        del delta_tensor, mask
        torch.cuda.empty_cache()

    return delta_dropped

def ties_merge(deltas: list, role_drop_rates_list: list, k, k_percent=50):
    """
    deltas: list of {key: tensor}
    role_drop_rates_list: list of {role: {key: drop_rate}}
    k: key
    k_percent: trim rate

    merged_ties: {key: tensor}
    """
    merged_ties = {}
    device = torch.device("cuda")

    # 1. Trim
    # Flatten and stack delta tensor of fine-tuned models
    stacked = torch.stack([d[k].flatten() for d in deltas], dim=0)  # shape: [num_models, N]

    # Move to GPU
    stacked = stacked.to(device)
    abs_stacked = torch.abs(stacked)

    # Calculate top-k%
    num_elements = stacked.numel()
    topk = int(num_elements * k_percent / 100)

    # Find top-k% threshold
    all_abs = abs_stacked.view(-1)
    thresh, _ = torch.kthvalue(all_abs, len(all_abs) - topk)

    # Create mask
    mask = (abs_stacked >= thresh).to(device)
    trimmed = stacked * mask.float()

    # 2. Elect sign
    pos_sum = torch.zeros_like(trimmed[0], device=device)
    pos_count = torch.zeros_like(trimmed[0], device=device)
    neg_sum = torch.zeros_like(trimmed[0], device=device)
    neg_count = torch.zeros_like(trimmed[0], device=device)

    for t in trimmed:
        pos_mask = (t > 0).float().to(device)
        neg_mask = (t < 0).float().to(device)
        pos_sum += t * pos_mask.float()
        pos_count += pos_mask.float()
        neg_sum += t * neg_mask.float()
        neg_count += neg_mask.float()

    # 3. Disjoint Merge
    merged = torch.zeros_like(pos_sum)

    # Compare abs of pos/neg
    choose_pos = torch.abs(pos_sum) > torch.abs(neg_sum)
    choose_neg = torch.abs(neg_sum) > torch.abs(pos_sum)
    equal_abs = torch.abs(pos_sum) == torch.abs(neg_sum)  # 절댓값 동일

    merged[choose_pos] = pos_sum[choose_pos] / pos_count[choose_pos].clamp(min=1)
    merged[choose_neg] = neg_sum[choose_neg] / neg_count[choose_neg].clamp(min=1)

    # 4. Rescaling with smallest drop rate
    role = get_role(k)
    if role is not None:
        min_drop = min([rd.get(role, {}).get(k, 0.0) for rd in role_drop_rates_list])
        merged *= 1.0 / (1.0 - min_drop)

    merged = merged.to("cpu")   # Save results to CPU

    # Reshape to original
    merged_ties[k] = merged.view(deltas[0][k].shape)

    del stacked, abs_stacked, all_abs, mask, trimmed
    del pos_mask, neg_mask, pos_sum, pos_count, neg_sum, neg_count
    del merged, choose_pos, choose_neg, equal_abs
    torch.cuda.empty_cache()

    return merged_ties

def simple_average(deltas: list, role_drop_rates_list: list, k):
    """
    deltas: list of {key: tensor}
    role_drop_rates_list: list of {role: {key: drop_rate}}
    k: key

    merged_simple: {key: tensor}
    """
    merged_simple = {}
    device = torch.device("cuda")

    # 1. Elect sign
    # Flatten and stack delta tensor of fine-tuned models
    stacked = torch.stack([d[k].flatten() for d in deltas], dim=0)  # shape: [num_models, N]

    # Move to GPU
    stacked = stacked.to(device)
    pos_sum = torch.zeros_like(stacked[0], device=device)
    pos_count = torch.zeros_like(stacked[0], device=device)
    neg_sum = torch.zeros_like(stacked[0], device=device)
    neg_count = torch.zeros_like(stacked[0], device=device)

    for t in stacked:
        pos_mask = (t > 0).float().to(device)
        neg_mask = (t < 0).float().to(device)
        pos_sum += t * pos_mask.float()
        pos_count += pos_mask.float()
        neg_sum += t * neg_mask.float()
        neg_count += neg_mask.float()

    # 2. Disjoint Merge
    merged = torch.zeros_like(pos_sum)

    # Compare abs of pos/neg
    choose_pos = torch.abs(pos_sum) > torch.abs(neg_sum)
    choose_neg = torch.abs(neg_sum) > torch.abs(pos_sum)
    equal_abs = torch.abs(pos_sum) == torch.abs(neg_sum)  # 절댓값 동일

    merged[choose_pos] = pos_sum[choose_pos] / pos_count[choose_pos].clamp(min=1)
    merged[choose_neg] = neg_sum[choose_neg] / neg_count[choose_neg].clamp(min=1)

    # 3. Rescaling with smallest drop rate
    role = get_role(k)
    if role is not None:
        min_drop = min([rd.get(role, {}).get(k, 0.0) for rd in role_drop_rates_list])
        merged *= 1.0 / (1.0 - min_drop)

    merged = merged.to("cpu")   # Save results to CPU

    # Reshape to original
    merged_simple[k] = merged.view(deltas[0][k].shape)

    del stacked
    del pos_mask, neg_mask, pos_sum, pos_count, neg_sum, neg_count
    del merged, choose_pos, choose_neg, equal_abs
    torch.cuda.empty_cache()

    return merged_simple

def merge_with_rms_filter(deltas: list, role_drop_rates_list: list, role_rms_list: list, topk_percent=10):
    """
    deltas: list of {key: tensor}
    role_drop_rates_list: list of {role: {key: drop_rate}}
    role_rms_list: list of role_rms: {role: {key: rms}}
    topk_percent: top-k% RMS values, keys containing these RMS go into ties-merging

    merged_delta: {key: tensor}
    """

    # 1. 모든 모델 RMS flatten
    all_rms = []
    all_keys_for_rms = []
    for role_rms in role_rms_list:
        for _, rms_dict in role_rms.items():
            for k, rms_val in rms_dict.items():
                all_rms.append(rms_val)
                all_keys_for_rms.append(k)

    # 2. Top-k RMS 임계값 계산
    all_rms_tensor = torch.tensor(all_rms)
    num_topk_rms = int(len(all_rms_tensor) * topk_percent / 100)
    if num_topk_rms == 0:
        topk_keys = set()
    else:
        # RMS 값 내림차순 정렬 후 상위 num_topk_rms 선택
        thresh, _ = torch.kthvalue(all_rms_tensor, len(all_rms_tensor) - num_topk_rms)
        topk_keys = {k for k, rms_val in zip(all_keys_for_rms, all_rms) if rms_val >= thresh}

    # 3. Remaining keys
    all_unique_keys = set(all_keys_for_rms)
    remaining_keys = all_unique_keys - topk_keys

    # 4. 비율 출력
    print(f"Total keys: {len(all_unique_keys)}")
    print(f"Top-k keys: {len(topk_keys)} ({len(topk_keys)/len(all_unique_keys)*100:.2f}%)")
    print(f"Remaining keys: {len(remaining_keys)} ({len(remaining_keys)/len(all_unique_keys)*100:.2f}%)")

    # 5. Top-k keys ties-merging
    merged_delta = {}
    if topk_keys:
        for k in topk_keys:
            merged_ties = ties_merge(deltas, role_drop_rates_list, k, k_percent=50)
            merged_delta[k] = merged_ties[k]

    # 6. Other keys simple average
    if remaining_keys:
        for k in remaining_keys:
            merged_simple = simple_average(deltas, role_drop_rates_list, k)
            merged_delta[k] = merged_simple[k]

    return merged_delta




################################################
### 1. Compute role rms of fine-tuned models ###
################################################

role_rms1, role_minmax1, delta1 = compute_role_rms(base_model_name, finetuned_model_name1, device)
role_rms2, role_minmax2, delta2 = compute_role_rms(base_model_name, finetuned_model_name2, device)
role_rms3, role_minmax3, delta3 = compute_role_rms(base_model_name, finetuned_model_name3, device)

"""
for k, v in delta1.items():
    print(k, v.shape, v.device)
print(f"\n")
"""

# Print role rms
# Fine-tuned model 1
print(f"=== role rms of ", {finetuned_model_name1}, " ===\n")
for role, rms in role_rms1.items():
    print(f"=== {role} ===")
    for i, (k, v) in enumerate(rms.items()):
        if i >= 10:
            break
        print(f"{k}: {v:.4f}")
print(f"===================================================\n")
"""
# Fine-tuned model 2
print(f"=== role rms of ", {finetuned_model_name2}, " ===\n")
for role, rms in role_rms2.items():
    print(f"=== {role} ===")
    for i, (k, v) in enumerate(rms.items()):
        if i >= 10:
            break
        print(f"{k}: {v:.4f}")
print(f"===================================================\n")

# Fine-tuned model 3
print(f"=== role rms of ", {finetuned_model_name3}, " ===\n")
for role, rms in role_rms3.items():
    print(f"=== {role} ===")
    for i, (k, v) in enumerate(rms.items()):
        if i >= 10:
            break
        print(f"{k}: {v:.4f}")
print(f"===================================================\n")
"""

# Print role min/max/mean
# Fine-tuned model 1
print(f"=== role min/max/mean of ", {finetuned_model_name1}, " ===\n")
for role, (rmin, rmax, rmean) in role_minmax1.items():
    print(f"{role}: min={rmin:.4f}, max={rmax:.4f}, mean={rmean:.4f}")
print(f"===================================================\n")

# Fine-tuned model 2
print(f"=== role min/max/mean of ", {finetuned_model_name2}, " ===\n")
for role, (rmin, rmax, rmean) in role_minmax2.items():
    print(f"{role}: min={rmin:.4f}, max={rmax:.4f}, mean={rmean:.4f}")
print(f"===================================================\n")

# Fine-tuned model 3
print(f"=== role min/max/mean of ", {finetuned_model_name3}, " ===\n")
for role, (rmin, rmax, rmean) in role_minmax3.items():
    print(f"{role}: min={rmin:.4f}, max={rmax:.4f}, mean={rmean:.4f}")
print(f"===================================================\n")




############################################################
### 2. Combine role rms and min/max/mean of fine-tuned models ###
############################################################

# role_minmax_list: Combined role_minmax dictionary of fine-tuned models
role_minmax_list = [role_minmax1, role_minmax2, role_minmax3]

# Combine role min/max/mean of fine-tuned models
all_roles = set()
for rm in role_minmax_list:
    all_roles.update(rm.keys())

# Total min/max/mean of each role
combined_minmax = {}
for role in all_roles:
    rmin = float('inf')
    rmax = float('-inf')
    rmean = 0.0
    for rm in role_minmax_list:
        if role in rm:
            rmin = min(rmin, rm[role][0])
            rmax = max(rmax, rm[role][1])
            rmean += rm[role][2]
    combined_minmax[role] = (rmin, rmax, rmean / len(role_minmax_list))

# Print combined min/max/mean rms
print(f"\n===================================================\n")
print(f"=== combined role min/max/mean  ===\n")
for role, (rmin, rmax, rmean) in combined_minmax.items():
    print(f"{role}: min={rmin:.4f}, max={rmax:.4f}, mean={rmean:.4f}")
print(f"===================================================\n")




##################################################
### 3. Compute role scaled and role drop rates ###
##################################################

# Compute role drop rates
role_drop_rates1 = compute_role_drop_rates(delta1, role_rms1, combined_minmax)
role_drop_rates2 = compute_role_drop_rates(delta2, role_rms2, combined_minmax)
role_drop_rates3 = compute_role_drop_rates(delta3, role_rms3, combined_minmax)




###########################
### 4. Apply delta drop ###
###########################

delta_dropped1 = apply_delta_drop(delta1, role_drop_rates1)
delta_dropped2 = apply_delta_drop(delta2, role_drop_rates2)
delta_dropped3 = apply_delta_drop(delta3, role_drop_rates3)

print(f"\n===================================================\n")
print(f"=== apply delta drop of ", {finetuned_model_name1}, " ===\n")
for i, k in enumerate(delta_dropped1.keys()):
    orig_tensor1 = delta1[k]
    dropped_tensor1 = delta_dropped1[k]

    # 원본 delta에서 0인 비율
    orig_zero_ratio1 = (orig_tensor1 == 0).sum().item() / orig_tensor1.numel()

    # drop 적용 후 0인 비율
    dropped_zero_ratio1 = (dropped_tensor1 == 0).sum().item() / dropped_tensor1.numel()

    # drop rate
    role = get_role(k)
    if role is None:
        continue
    drop_rate1 = role_drop_rates1[role][k]
    
    print(f"{k}: before_drop={orig_zero_ratio1:.4f}, after_drop={dropped_zero_ratio1:.4f}, drop_rate={drop_rate1:.4f}")

del delta1, orig_tensor1, dropped_tensor1, drop_rate1
print(f"===================================================\n")

print(f"=== apply delta drop of ", {finetuned_model_name2}, " ===\n")
for i, k in enumerate(delta_dropped2.keys()):
    orig_tensor2 = delta2[k]
    dropped_tensor2 = delta_dropped2[k]

    # 원본 delta에서 0인 비율
    orig_zero_ratio2 = (orig_tensor2 == 0).sum().item() / orig_tensor2.numel()

    # drop 적용 후 0인 비율
    dropped_zero_ratio2 = (dropped_tensor2 == 0).sum().item() / dropped_tensor2.numel()

    # drop rate
    role = get_role(k)
    if role is None:
        continue
    drop_rate2 = role_drop_rates2[role][k]

    print(f"{k}: orig_zero={orig_zero_ratio2:.4f}, after_drop={dropped_zero_ratio2:.4f}, drop_rate={drop_rate2:.4f}")

del delta2, orig_tensor2, dropped_tensor2, drop_rate2
print(f"===================================================\n")

print(f"=== apply delta drop of ", {finetuned_model_name3}, " ===\n")
for i, k in enumerate(delta_dropped3.keys()):
    orig_tensor3 = delta3[k]
    dropped_tensor3 = delta_dropped3[k]

    # 원본 delta에서 0인 비율
    orig_zero_ratio3 = (orig_tensor3 == 0).sum().item() / orig_tensor3.numel()

    # drop 적용 후 0인 비율
    dropped_zero_ratio3 = (dropped_tensor3 == 0).sum().item() / dropped_tensor3.numel()

    # drop rate
    role = get_role(k)
    if role is None:
        continue
    drop_rate3 = role_drop_rates3[role][k]

    print(f"{k}: orig_zero={orig_zero_ratio3:.4f}, after_drop={dropped_zero_ratio3:.4f}, drop_rate={drop_rate3:.4f}")

del delta3, orig_tensor3, dropped_tensor3, drop_rate3
print(f"===================================================\n")




###############################
### 5. Merge and save model ###
###############################

token = None  # 필요시 Hugging Face 토큰 입력
device = "cpu"  # 디버깅 시 CPU 사용

# -----------------------------
# delta 병합
# -----------------------------
# 기존 delta_dropped*, role_drop_rates*, role_rms* 사용
merged_delta = merge_with_rms_filter(
    [delta_dropped1, delta_dropped2, delta_dropped3],
    [role_drop_rates1, role_drop_rates2, role_drop_rates3],
    [role_rms1, role_rms2, role_rms3]
)


# 1️⃣ Base 모델 로드 (CPU, float16)
merged_model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,  # float16으로 변경
    device_map="cpu",
    low_cpu_mem_usage=True,
    token=token
)

# 2️⃣ Delta 병합
state_dict = merged_model.state_dict()
for k, delta in merged_delta.items():
    if k in state_dict:
        # delta도 float16으로 맞춰서 합산
        state_dict[k] += delta.to(state_dict[k].device, dtype=torch.float16)
    del delta
torch.cuda.empty_cache()  # GPU 사용 시 캐시 정리

# 3️⃣ State 덮어쓰기
merged_model.load_state_dict(state_dict)
merged_model.eval()

# 4️⃣ 모델 저장
merged_model.save_pretrained(save_path)
tokenizer = LlamaTokenizer.from_pretrained(base_model_name, token=token)
tokenizer.save_pretrained(save_path)

print(f"✅ Merged model saved to {save_path}")
