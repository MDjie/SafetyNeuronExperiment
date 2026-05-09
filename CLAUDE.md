# SafetyNeuron Project Context

## Paper
"Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons" (NeurIPS 2025)

## Current Task: Dynamic Activation Patching Experiment

Experiment directory: `experiments/llama2_patching/`

### What this experiment does
Compares base model (Llama-2-7b-hf) vs chat model (Llama-2-7b-chat-hf) using the paper's **dynamic activation patching** (Algorithm 1):
- At each generation token step, the aligned (chat) model runs with `run_with_cache()`
- Its `hook_post` activations at safety-neuron positions replace the base model's activations in real-time
- Causal Score (Equation 4): `C = E[F(patched) - F(baseline)] / E[F(chat) - F(baseline)]`
- Scoring via cost model (beaver-7b-v1.0-cost) with regex fallback

### Experiment status
- **200 samples** from hh_rlhf_harmless.jsonl
- **8 topk values**: 3400, 6800, 10200, 13600, 17000, 20400, 23800, 27200 (1%-8% of neurons)
- **Completed**: baseline, chat_reference, topk=3400, 6800, 10200 (both chat and random)
- **Remaining**: topk=13600, 17000, 20400, 23800, 27200

### Key files
- `run_patching.py` — Main experiment script (rewritten for dynamic patching + cost model scoring)
- `instruction.sh` — Runnable command to resume experiment
- `collect_activations.py` — Stage 2: pre-compute activation stats
- `outputs/neuron_activation.pt` — Combined Stage 1 data (change_scores, neuron_ranks, base/chat mean/std)
- `outputs/baseline.jsonl` — 200 baseline completions (no patching)
- `outputs/chat_reference.jsonl` — 200 chat model reference completions
- `outputs/patch_chat_top{topk}.jsonl` — Safety neuron patching completions
- `outputs/patch_random_top{topk}.jsonl` — Random neuron control completions
- `outputs/causal_summary.json` — Final causal scores
- `outputs/causal_effect_plot.png` — Line chart

### Key code features in run_patching.py
- `aggressive_cleanup()` — gc.collect() + CUDA synchronize/empty_cache/ipc_collect between model loads
- `load_models_with_retry()` — Up to 2 retries on OOM with cleanup + 3s wait
- `_load_existing_jsonl()` — Auto-detects complete JSONL files and loads from disk (skips regeneration)
- `--skip_reference` flag — Skips baseline & chat reference generation if files exist
- Dynamic patching via `generate_with_dynamic_patching()` using `guided_model` + `hook_fn`

### Known issues & fixes
- **GPU OOM from memory fragmentation**: Repeated model load/unload cycles fragment GPU memory. Fixed with aggressive_cleanup() + retry logic.
- **Repetitive generation**: Greedy decoding (do_sample=False) causes repetition. Fixed with `repetition_penalty=1.1` (works in greedy mode; temperature does NOT work in greedy mode).
- **Plot showing zeros**: Key mismatch — `patch_chat_top{topk}` vs `chat_top{topk}`. Fixed.
- **Process stuck on closed terminal**: stdout goes to pty; if terminal disconnects, tqdm/print blocks. Run with `nohup` or in `screen`/`tmux`.

### Resuming the experiment
```bash
cd /mnt/workspace/SafetyNeuron/experiments/llama2_patching
bash instruction.sh
```
This uses `--skip_reference` and auto-loads completed topk values from disk.

### Dependencies (beyond Alibaba DWS base)
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    transformers==4.39.3 peft==0.7.0 accelerate==0.29.1 \
    datasets==2.15.0 einops==0.7.0 jaxtyping==0.2.24 \
    safetensors==0.4.1 transformer-lens==1.11.0 bitsandbytes

pip install matplotlib -i https://mirrors.huaweicloud.com/repository/pypi/simple
pip install addict modelscope
```
Note: tokenizers==0.15.0 does NOT exist on mirrors; transformers 4.39.3 auto-installs tokenizers==0.15.2.

### Models needed
- `./models/Llama-2-7b-hf` — Base model
- `./models/Llama-2-7b-chat-hf` — Chat/aligned model
- `./models/PKU-Alignment/beaver-7b-v1.0-cost` — Cost model (downloaded from ModelScope, not HuggingFace)

Download cost model:
```bash
python -c "from modelscope import snapshot_download; snapshot_download('PKU-Alignment/beaver-7b-v1.0-cost', cache_dir='./models')"
```

### Architecture notes
- Guided generation path: `generate()` → `_prepare_generation_config()` → `_validate_model_kwargs()` → `_greedy_search(**model_kwargs)`
- `guided_model` parameter passed through `model_kwargs` in HookedModelBase.py
- Dynamic patching happens in `_greedy_search()` lines 1119-1154: `chat_model.run_with_cache()` → extract activations → patch base model
- 8-bit quantization needed for fitting two 7B models on 24GB GPU
- neuron_activation.pt shape: 32 layers × 11008 neurons
