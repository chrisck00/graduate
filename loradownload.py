from huggingface_hub import snapshot_download
dataset="piqa"
snapshot_download(
    repo_id=f"Styxxxx/llama2_7b_lora-{dataset}",
    local_dir=f"./lora/{dataset}",
    local_dir_use_symlinks=False
)
#["rte","boolq","hellaswag","arc_challenge", "openbookqa","piqa","coqa"]
# python loradownload.py