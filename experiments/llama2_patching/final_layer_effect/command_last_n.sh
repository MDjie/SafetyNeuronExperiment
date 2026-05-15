#!/bin/bash
# Progressive last-n-layer patching experiments.
# Tests how causal effect grows as we patch neurons from more layers (in reverse).
#
# n=1:  layer 30 only       (11,008 neurons)
# n=2:  layers 29-30        (22,016)
# n=3:  layers 28-30        (33,024)
# n=4:  layers 27-30        (44,032)
# n=5:  layers 26-30        (55,040)
# n=7:  layers 24-30        (77,056)
# n=10: layers 21-30        (110,080)
# n=15: layers 16-30        (165,120)
# n=31: all layers 0-30     (341,248)
#
# Reference completions (baseline/chat_reference) are read from ../outputs/
# (shared with the original run_layer_effect.py).

python run_last_n_layers.py \
    --base_model /mnt/workspace/models/Llama-2-7b-hf \
    --chat_model /mnt/workspace/models/Llama-2-7b-chat-hf \
    --prompt_file ../data/hh_rlhf_harmless.jsonl \
    --stage1_dir ../../llama2_safety_neurons/outputs \
    --output_dir ./outputs/last_n_layers \
    --reference_dir ./outputs \
    --cost_model /mnt/workspace/models/beaver-7b-v1.0-cost \
    --load_in_8bit \
    --num_samples 200 --batch_size 48 --max_new_tokens 128 \
    --last_n_list 1 2 3 4 5 7 10 15 31
