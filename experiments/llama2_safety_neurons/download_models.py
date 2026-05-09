"""
Download Llama-2-7b-hf and Llama-2-7b-chat-hf from ModelScope.

Usage:
    # Download both models with default ModelScope IDs
    python download_models.py --target_dir ./models

    # Download only base model
    python download_models.py --target_dir ./models --model base

    # Override specific model ID (if default doesn't work)
    python download_models.py --target_dir ./models --base_id "your-org/Llama-2-7b-hf"

    # Download from HuggingFace instead (if you have access)
    python download_models.py --source hf --target_dir ./models
"""

import os
import argparse

# Default ModelScope IDs — verified to exist on modelscope.cn
MODELSCOPE_IDS = {
    "base": "vllm-ascend/Llama-2-7b-hf",
    "chat": "shakechen/Llama-2-7b-chat-hf",
}

# HuggingFace IDs — requires Meta's license approval
HF_IDS = {
    "base": "meta-llama/Llama-2-7b-hf",
    "chat": "meta-llama/Llama-2-7b-chat-hf",
}


def download_from_modelscope(model_key: str, model_id: str, target_dir: str):
    from modelscope import snapshot_download

    local_dir = os.path.join(target_dir, model_id.split("/")[-1])
    print(f"[{model_key}] ModelScope: {model_id} -> {local_dir}")
    snapshot_download(model_id=model_id, local_dir=local_dir)
    print(f"[{model_key}] Done -> {local_dir}")
    return local_dir


def download_from_huggingface(model_key: str, model_id: str, target_dir: str):
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(target_dir, model_id.split("/")[-1])
    print(f"[{model_key}] HuggingFace: {model_id} -> {local_dir}")
    snapshot_download(repo_id=model_id, local_dir=local_dir,
                      local_dir_use_symlinks=False)
    print(f"[{model_key}] Done -> {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download Llama-2 models from ModelScope or HuggingFace")
    parser.add_argument("--target_dir", type=str, default="./models",
                        help="Directory to store downloaded models.")
    parser.add_argument("--model", type=str, choices=["base", "chat", "both"],
                        default="both")
    parser.add_argument("--source", type=str, choices=["ms", "hf"],
                        default="ms",
                        help="Download source: ms = ModelScope, hf = HuggingFace")
    parser.add_argument("--base_id", type=str, default=None,
                        help="Override default base model ID.")
    parser.add_argument("--chat_id", type=str, default=None,
                        help="Override default chat model ID.")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    ids = MODELSCOPE_IDS if args.source == "ms" else HF_IDS
    if args.base_id:
        ids["base"] = args.base_id
    if args.chat_id:
        ids["chat"] = args.chat_id

    download_fn = (download_from_modelscope if args.source == "ms"
                   else download_from_huggingface)

    print(f"Source: {'ModelScope' if args.source == 'ms' else 'HuggingFace'}")
    print(f"Base model ID: {ids['base']}")
    print(f"Chat model ID: {ids['chat']}")
    print()

    if args.model in ("base", "both"):
        download_fn("base", ids["base"], args.target_dir)
    if args.model in ("chat", "both"):
        download_fn("chat", ids["chat"], args.target_dir)


if __name__ == "__main__":
    main()
