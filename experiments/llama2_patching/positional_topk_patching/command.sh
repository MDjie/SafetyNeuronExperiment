python run_positional_topk.py \
    --base_model /mnt/workspace/models/Llama-2-7b-hf \
    --chat_model /mnt/workspace/models/Llama-2-7b-chat-hf \
    --prompt_file ../data/hh_rlhf_harmless.jsonl \
    --stage1_dir ../../llama2_safety_neurons/outputs \
    --output_dir ./outputs \
    --cost_model /mnt/workspace/models/beaver-7b-v1.0-cost \
    --load_in_8bit \
    --num_samples 200 --batch_size 48 --max_new_tokens 128
