from huggingface_hub import snapshot_download
dataset=["rte","bool_q","hellaswag","arc_challenge", "openbookqa","piqa","coqa","glue_qqp"]
for data in dataset:
    snapshot_download(
        repo_id=f"Styxxxx/llama2_7b_lora-{data}",
        local_dir=f"./lora/{data}",
        local_dir_use_symlinks=False
    )
snapshot_download(
    repo_id=f"meta-llama/Llama-2-7b-hf",
    local_dir=f"./base_model",
    local_dir_use_symlinks=False
)
#["rte","boolq","hellaswag","arc_challenge", "openbookqa","piqa","coqa"]
# python loradownload.py
