"""
Stage 2: Collect per-neuron activation statistics for safety neuron patching.

This script computes mean/std activations of every MLP post-activation neuron
on harmful prompts (at the last prompt token), for both the base and chat models,
then bundles them with change scores and neuron ranks from Stage 1.

Output format (compatible with eval.arena.run_eval):
    (change_scores, neuron_ranks, base_mean, base_std, chat_mean, chat_std)

Usage:
    # Full collection on 100 prompts (recommended)
    python collect_activations.py \
        --base_model ./models/Llama-2-7b-hf \
        --chat_model ./models/Llama-2-7b-chat-hf \
        --prompt_file ./data/hh_rlhf_harmless.jsonl \
        --stage1_dir ../llama2_safety_neurons/outputs/chat_completions \
        --output_dir ./outputs \
        --num_samples 100
"""

import os
import sys
import json
import argparse
import random

import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import seed_torch
from activation_processor import BaseActivationProcessor
from eval.utils import load_hooked_lm_and_tokenizer
from eval.templates import create_prompt_with_llama2_chat_format


def load_prompts(file_path: str, num_samples: int) -> list[str]:
    with open(file_path, "r") as f:
        data = [json.loads(line)["prompt"].strip() for line in f if line.strip()]
    print(f"Loaded {len(data)} prompts from {file_path}")
    if 0 < num_samples < len(data):
        data = random.Random(42).sample(data, num_samples)
    print(f"Using {len(data)} prompts.")
    return data


def collect_activation_stats(model, tokenizer, prompts: list[str],
                             processor: BaseActivationProcessor,
                             names_filter) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run model with cache on each prompt (prompt_last token), accumulate
    per-neuron mean and std across all prompts.

    Returns (mean, std), each of shape [n_layers, n_neurons].
    """
    device = model.device
    sum_x = None
    sum_x2 = None
    n_total = 0

    for i in tqdm(range(0, len(prompts), processor.batchsize), desc="Collecting activations"):
        batch = prompts[i:i + processor.batchsize]
        # token_type='prompt_last' — only the last prompt token (no generation)
        batch_ids, batch_masks, batch_select = processor.process_prompts(
            model, batch, token_type="prompt_last")

        activation = processor._get_activation(
            model, batch_ids, batch_masks, batch_select, names_filter=names_filter)
        # activation: [n_tokens, n_layers, n_neurons]

        if sum_x is None:
            sum_x = activation.sum(dim=0)
            sum_x2 = (activation * activation).sum(dim=0)
        else:
            sum_x += activation.sum(dim=0)
            sum_x2 += (activation * activation).sum(dim=0)
        n_total += activation.shape[0]

        del activation

    mean = sum_x / n_total
    # E[X^2] - (E[X])^2
    variance = (sum_x2 / n_total) - (mean * mean)
    std = variance.clamp(min=0).sqrt()
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-neuron activation statistics for safety neuron patching")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--chat_model", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--stage1_dir", type=str, required=True,
                        help="Path to Stage 1 outputs (contains change_scores.pt etc.).")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--output_name", type=str, default="neuron_activation",
                        help="Output filename stem.")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    args = parser.parse_args()

    seed_torch(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU memory: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

    load_kwargs = {
        "device_map": "auto",
        "load_in_8bit": args.load_in_8bit,
        "torch_dtype": torch.float16,
    }
    names_filter = lambda name: name.endswith("hook_post") and "31" not in name

    # Load Stage 1 change scores and neuron ranks
    print("\n=== Loading Stage 1 results ===")
    change_scores = torch.load(os.path.join(args.stage1_dir, "change_scores.pt"), map_location="cpu")
    neuron_ranks = torch.load(os.path.join(args.stage1_dir, "neuron_ranks.pt"), map_location="cpu")
    print(f"  change_scores: {tuple(change_scores.shape)}")
    print(f"  neuron_ranks:  {tuple(neuron_ranks.shape)}")

    # Load prompts
    raw_prompts = load_prompts(args.prompt_file, args.num_samples)
    chat_prompts = []
    for p in raw_prompts:
        messages = [{"role": "user", "content": p}]
        chat_prompts.append(create_prompt_with_llama2_chat_format(messages, add_bos=False))

    processor = BaseActivationProcessor(batchsize=args.batch_size)

    # Collect base model activation stats
    print("\n=== Collecting base model activations (prompt_last) ===")
    base_model, base_tokenizer = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.base_model,
        tokenizer_name_or_path=args.base_model,
        peft_name_or_path=None,
        **load_kwargs,
    )
    base_model.set_tokenizer(base_tokenizer)
    base_mean, base_std = collect_activation_stats(
        base_model, base_tokenizer, chat_prompts, processor, names_filter)
    print(f"  base_mean: {tuple(base_mean.shape)}, base_std: {tuple(base_std.shape)}")
    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    # Collect chat model activation stats
    print("\n=== Collecting chat model activations (prompt_last) ===")
    chat_model, chat_tokenizer = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.chat_model,
        tokenizer_name_or_path=args.chat_model,
        peft_name_or_path=None,
        **load_kwargs,
    )
    chat_model.set_tokenizer(chat_tokenizer)
    chat_mean, chat_std = collect_activation_stats(
        chat_model, chat_tokenizer, chat_prompts, processor, names_filter)
    print(f"  chat_mean: {tuple(chat_mean.shape)}, chat_std: {tuple(chat_std.shape)}")
    del chat_model, chat_tokenizer
    torch.cuda.empty_cache()

    # Save in the format expected by eval.arena.run_eval:
    # (change_scores, neuron_ranks, base_mean, base_std, chat_mean, chat_std)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.pt")
    torch.save(
        (change_scores, neuron_ranks, base_mean.cpu(), base_std.cpu(),
         chat_mean.cpu(), chat_std.cpu()),
        output_path,
    )
    print(f"\nSaved -> {output_path}")
    print("Ready for Stage 3: run_patching.py")


if __name__ == "__main__":
    main()
