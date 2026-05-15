#!/bin/bash
# Per-layer top-k dose-response sweep.
# Gradually increases the fraction of top neurons per layer,
# from 1/15 → 15/15 (all neurons) across the last 15 layers (16-30).
#
# DOSE-RESPONSE TABLE (expected):
#   frac  topk/layer  total_neurons
#   0.07       733       10,995      ≈ same as layer-30-only (11,008)
#   0.13      1,467      22,005
#   0.20      2,201      33,015
#   0.33      3,669      55,035
#   0.47      5,137      77,055
#   0.67      7,339     110,085
#   1.00     11,008     165,120      ← all neurons from last 15 layers
#
# KEY COMPARISONS:
#   Layer 30 only (11,008 neurons, 1 layer)  → CE ≈ -0.023
#   Last 15, top 1/15 per layer (10,995)     → ???
#   Last 15, all neurons (165,120)            → ???
#
# If CE rises sharply early and plateaus, safety signal is concentrated
# in top per-layer neurons → sparse distributed circuit.
# If CE rises slowly/linearly, safety requires dense per-layer activation.

python run_per_layer_topk.py \
    --base_model /mnt/workspace/models/Llama-2-7b-hf \
    --chat_model /mnt/workspace/models/Llama-2-7b-chat-hf \
    --prompt_file ../data/hh_rlhf_harmless.jsonl \
    --stage1_dir ../../llama2_safety_neurons/outputs \
    --output_dir ./outputs/per_layer_topk \
    --reference_dir ./outputs \
    --cost_model /mnt/workspace/models/beaver-7b-v1.0-cost \
    --load_in_8bit \
    --num_samples 200 --batch_size 48 --max_new_tokens 128 \
    --last_n 15 \
    --per_layer_frac_list 0.0667 0.133 0.2 0.333 0.467 0.667 1.0
