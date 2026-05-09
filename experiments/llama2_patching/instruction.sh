# Resume experiment with memory-safe retry logic.
# --skip_reference: baseline & chat reference already exist.
# Completed topk (3400,6800,10200) auto-loaded from disk; only 13600+ will run.
python3 run_patching.py --base_model ./models/Llama-2-7b-hf --chat_model ./models/Llama-2-7b-chat-hf --prompt_file ./data/hh_rlhf_harmless.jsonl --index_path ./outputs/neuron_activation.pt --output_dir ./outputs --topk 3400 6800 10200 13600 17000 20400 23800 27200 --num_samples 200 --load_in_8bit --cost_model ./models/PKU-Alignment/beaver-7b-v1.0-cost --max_new_tokens 128 --seed 42 --skip_reference
