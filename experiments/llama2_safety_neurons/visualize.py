"""
Generate two visualization charts for safety neuron analysis results.

Chart 1: Neuron location distribution — top-K safety neuron count per layer.
Chart 2: Average activation by layer — base vs chat model activation magnitudes.
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIRS = {
    "Chat Completions": "./outputs/chat_completions",
    "Base Completions": "./outputs/base_completions",
}
TOP_K = 10000


def load_tensor(path, device="cpu"):
    return torch.load(os.path.join(path), map_location=device)


def plot_neuron_distribution(ax, output_dir, label, top_k=TOP_K):
    """Bar chart: number of top-K safety neurons per layer."""
    neuron_ranks = load_tensor(os.path.join(output_dir, "neuron_ranks.pt"))
    layers = neuron_ranks[:top_k, 0].tolist()
    layer_ids = sorted(set(layers))
    counts = [layers.count(l) for l in layer_ids]
    ax.bar(layer_ids, counts, alpha=0.75)
    ax.set_title(f"Top-{top_k} Safety Neuron Distribution ({label})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Neuron Count")
    ax.set_xticks(layer_ids)
    ax.set_xticklabels(layer_ids, rotation=45, fontsize=7)


def plot_avg_activation_by_layer(ax, output_dir, label):
    """Grouped bar: mean activation magnitude per layer (base vs chat)."""
    first_mean = load_tensor(os.path.join(output_dir, "first_mean.pt"))   # base
    second_mean = load_tensor(os.path.join(output_dir, "second_mean.pt")) # chat

    n_layers = first_mean.shape[0]
    layers = list(range(n_layers))

    base_avg = first_mean.abs().mean(dim=1).tolist()
    chat_avg = second_mean.abs().mean(dim=1).tolist()

    x = range(n_layers)
    width = 0.35
    ax.bar([i - width/2 for i in x], base_avg, width, label="Base Model", alpha=0.8)
    ax.bar([i + width/2 for i in x], chat_avg, width, label="Chat Model", alpha=0.8)
    ax.set_title(f"Average |Activation| by Layer ({label})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Absolute Activation")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, fontsize=7)
    ax.legend(fontsize=7)


def main():
    fig_dir = "./outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Safety Neuron Analysis — Llama-2-7b-hf vs Llama-2-7b-chat-hf",
                 fontsize=13, fontweight="bold", y=1.01)

    for col, (label, path) in enumerate(OUTPUT_DIRS.items()):
        plot_neuron_distribution(axes[0, col], path, label)
        plot_avg_activation_by_layer(axes[1, col], path, label)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "safety_neuron_analysis.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
