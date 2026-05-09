"""
Download and prepare safety-alignment datasets for safety neuron discovery.

Supported datasets:
    - hh-rlhf    : Anthropic hh-rlhf (harmless subset) — primary dataset in the paper
    - beavertails: PKU-Alignment/BeaverTails (harmful QA pairs)
    - harmfulqa  : Custom harmful QA data (if available locally)

Output format (JSONL):
    {"dataset": "...", "id": "...", "prompt": "harmful_user_query_text"}

Usage:
    # Download & prepare hh-rlhf harmless data (default)
    python prepare_data.py --dataset hh_rlhf --output_dir ./data

    # Prepare BeaverTails
    python prepare_data.py --dataset beavertails --output_dir ./data

    # Multiple datasets, limit samples
    python prepare_data.py --dataset hh_rlhf beavertails --num_samples 500 \
        --output_dir ./data
"""

import os
import sys
import json
import random
import argparse
from typing import Optional

# Add project src/ to path for importing templates
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.templates import create_prompt_with_llama2_chat_format


def extract_last_user_query(messages: list[dict]) -> str:
    """Extract the last user message from a conversation."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"].strip()
    return ""


def prepare_hh_rlhf(output_dir: str, num_samples: Optional[int] = None,
                    hf_endpoint: Optional[str] = None) -> str:
    """
    Download Anthropic hh-rlhf and extract harmful/safety prompts from the
    harmless subset.  Each prompt is the last user query in a conversation
    where the assistant gave a safety-aware (harmless) response.
    """
    from datasets import load_dataset

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"Using HF mirror: {hf_endpoint}")

    print("Downloading Anthropic hh-rlhf (harmless subset)...")
    dataset = load_dataset("Anthropic/hh-rlhf", "harmless-base", split="test")
    print(f"  Loaded {len(dataset)} examples")

    # Parse and extract user queries
    prompts = []
    for idx, example in enumerate(dataset):
        # hh-rlhf 'chosen' field: "Human: ...\n\nAssistant: ...\n\nHuman: ..."
        # Build messages from the raw text
        messages = []
        raw = example["chosen"]
        for turn in raw.split("\n\n"):
            turn = turn.strip()
            if not turn:
                continue
            if turn.startswith("Human: "):
                messages.append({"role": "user", "content": turn[7:]})
            elif turn.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": turn[11:]})

        # Take the last user query as the prompt
        query = extract_last_user_query(messages)
        if query:
            prompts.append(query)

    print(f"  Extracted {len(prompts)} user queries")

    if num_samples and num_samples < len(prompts):
        prompts = random.Random(42).sample(prompts, num_samples)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hh_rlhf_harmless.jsonl")
    with open(output_path, "w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"dataset": "hh_rlhf", "id": f"hh_{i}", "prompt": p}) + "\n")

    print(f"  Saved {len(prompts)} prompts -> {output_path}")
    return output_path


def prepare_beavertails(output_dir: str, num_samples: Optional[int] = None,
                        hf_endpoint: Optional[str] = None) -> str:
    """
    Download PKU-Alignment/BeaverTails and extract harmful prompts.
    Filters to only include unsafe/questionable prompts.
    """
    from datasets import load_dataset

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"Using HF mirror: {hf_endpoint}")

    print("Downloading PKU-Alignment/BeaverTails...")
    # BeaverTails has a 330k test split with is_safe annotation
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    print(f"  Loaded {len(dataset)} examples")

    prompts = []
    for idx, example in enumerate(dataset):
        # Only keep prompts flagged as potentially unsafe
        # (prompts where is_safe is False or ambiguous)
        if not example.get("is_safe", True):
            prompt = example.get("prompt", "").strip()
            if prompt:
                prompts.append(prompt)

    print(f"  Extracted {len(prompts)} unsafe prompts")

    if num_samples and num_samples < len(prompts):
        prompts = random.Random(42).sample(prompts, num_samples)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "beavertails.jsonl")
    with open(output_path, "w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"dataset": "beavertails", "id": f"bt_{i}", "prompt": p}) + "\n")

    print(f"  Saved {len(prompts)} prompts -> {output_path}")
    return output_path


DATASET_BUILDERS = {
    "hh_rlhf": prepare_hh_rlhf,
    "beavertails": prepare_beavertails,
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare safety-alignment datasets for safety neuron discovery")
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=["hh_rlhf"],
                        choices=list(DATASET_BUILDERS.keys()),
                        help="Which dataset(s) to prepare.")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to save prepared JSONL files.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap samples per dataset (None = all).")
    parser.add_argument("--hf_mirror", type=str, default="https://hf-mirror.com",
                        help="HuggingFace mirror endpoint (e.g. https://hf-mirror.com). "
                             "Set to empty string to use default HF.")
    args = parser.parse_args()

    hf_endpoint = args.hf_mirror or None

    for ds in args.dataset:
        print(f"\n{'='*50}")
        DATASET_BUILDERS[ds](args.output_dir, args.num_samples, hf_endpoint=hf_endpoint)

    print("\nDone. Ready to run: python run.py --prompt_file ./data/<dataset>.jsonl")


if __name__ == "__main__":
    main()
