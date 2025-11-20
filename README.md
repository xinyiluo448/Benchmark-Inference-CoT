## Benchmark-Inference-CoT

Quick start for SST-2 (dev split):
```bash
# prepare jsonl from HF SST-2 dev
python3 data_prep/prepare_sst2.py --split validation --out sst2_dev.jsonl

# run benchmark on the converted file
python bench_framework.py --task classification-sst2 --data sst2_dev.jsonl --save-csv sst2_run.csv
# or use the helper script (now forwards model/bench args):
bash scripts/run_sst2_dev.sh --model mistralai/Mistral-7B-Instruct-v0.2 --max-new-tokens 64
```

Lightweight LLM inference benchmarking scaffold. Loads HuggingFace models (default `mistralai/Mistral-7B-Instruct-v0.2`), runs built-in mini tasks or custom datasets, reports latency, tokens/s, and peak GPU memory, and can save per-sample CSV.

### Layout
- `bench_framework.py`: CLI entry, argument parsing and orchestration.
- `bench_runner/`:
  - `model.py`: model/tokenizer loading (optional bitsandbytes 4-bit).
  - `tasks.py`: built-in tasks plus dataset loading/sampling helpers.
  - `runner.py`: single-sample inference, timing, and CSV writing.
  - `metrics.py`: simple evaluators (exact/numeric/rouge selector, default mappings).
- `data_prep/prepare_sst2.py`: convert HF `glue/sst2` into jsonl.
- `data_prep/prepare_gsm8k.py`: convert HF `gsm8k` into jsonl.
- `data_prep/prepare_xsum.py`: convert HF `xsum` into jsonl.
- `scripts/`: convenience entrypoints (`run_sst2_dev.sh`, `run_gsm8k.sh`, `run_xsum.sh`, `run_custom.sh`, `run_matrix.py`).
- `staging/`: scratch folder for prepared data and outputs.

### Quick start (built-in toy task)
```bash
python bench_framework.py --task reasoning-gsm8k --max-new-tokens 64
```
Prints prompt/output plus latency, tokens/s, and peak GPU memory for each example.

### Using your own data
Accepted formats: jsonl/json/csv with fields:
- `prompt` (required)
- `reference` (optional)
- `id` / `name` (optional, for identification)

Example:
```bash
python bench_framework.py \
  --task classification-sst2 \
  --data sst2_dev.jsonl \
  --max-samples 500 \
  --shuffle-seed 42 \
  --save-csv run.csv
```

### Prepare SST-2 data
Convert HF SST-2 to jsonl:
```bash
python3 data_prep/prepare_sst2.py --split validation --out sst2_dev.jsonl
python3 data_prep/prepare_sst2.py --split train --max-samples 50000 --shuffle-seed 42 --out sst2_train_sample.jsonl
```

### Prepare GSM8K data
```bash
python3 data_prep/prepare_gsm8k.py --split train --max-samples 100 --shuffle-seed 42 --out gsm8k_train_100.jsonl
python3 data_prep/prepare_gsm8k.py --split test --out gsm8k_test.jsonl
```

### Prepare XSum data
> Note: XSum requires `datasets<3` or a script-free parquet source. Install via `pip install "datasets<3"` if needed.
```bash
python3 data_prep/prepare_xsum.py --split validation --max-samples 100 --shuffle-seed 42 --out xsum_val_100.jsonl
python3 data_prep/prepare_xsum.py --split test --out xsum_test.jsonl
```

### Key arguments
- `--model`: HF model id (default `mistralai/Mistral-7B-Instruct-v0.2`)
- `--task`: built-in task key (`classification-sst2`, `reasoning-gsm8k`, `summarization-xsum`); can be overridden by `--data`
- `--task-config`: json/yaml task spec (data_path/prompt_template/reference_field/metric)
- `--data`: path to jsonl/json/csv
- `--max-samples` / `--shuffle-seed`: sampling control
- `--max-new-tokens`: generation limit
- `--load-4bit`: enable bitsandbytes 4-bit (requires installation)
- `--bits`: quantization bits (4/8/16; 16 = no quantization)
- `--dtype`: weight dtype (float16/bfloat16/float32)
- `--bits-list` / `--dtype-list`: sweep multiple precisions (used in `run_matrix.py`)
- `--model-configs`: JSON list of preset runs (fields: model, bits, dtype, use_chat_template) to bypass the cross-product sweep
- `--save-csv`: path to save per-sample results
- `--save-outputs`: also persist prompt/output/reference in CSV (useful for offline metrics like ROUGE)
- `--eval-rouge`: compute ROUGE-1/2/L in-process (requires `rouge-score`; needs references and outputs)
- `--use-chat-template`: apply `tokenizer.chat_template` for chat models (recommended for Qwen/Llama chat variants)

### Notes
- Requires a working python/python3 env with `transformers`, `torch`, `datasets` (for data prep), and bitsandbytes if using 4-bit.
- End-to-end time depends on GPU, tokens/s, and sample count; try small runs via `--max-samples` first.
- CSV outputs now save metrics only by default (name, latency_s, tokens_generated, tokens_per_s, peak_mem_gb, reference, correct); prompts/outputs can be stored via `--save-outputs`. The CLI prints per-sample details and a summary (avg latency, throughput, peak mem, accuracy with numeric extraction for GSM8K or exact match on `reference`). ROUGE F1 can be computed inline via `--eval-rouge` if `rouge-score` is installed.
- SST-2 判分使用了轻量标签归一化（包含 positive/negative 关键字时视为对应标签），减少长句误判。

### Compare multiple models/tasks
Use `scripts/run_matrix.py` to sweep models x tasks and aggregate summaries:
```bash
PYTHONPATH=. python scripts/run_matrix.py \
  --models mistralai/Mistral-7B-Instruct-v0.2 Qwen/Qwen1.5-7B-Chat \
  --tasks classification-sst2 reasoning-gsm8k \
  --data-map data_map.json \
  --out-dir staging/matrix_runs \
  --max-new-tokens 128 \
  --bits-list 4 8 16 \
  --dtype-list float16 bfloat16 \
  --use-chat-template \
  --eval-rouge \
  --save-outputs
```
`data_map.json` example:
```json
{
  "classification-sst2": "staging/sst2_dev.jsonl",
  "reasoning-gsm8k": "staging/gsm8k_train_100.jsonl",
  "summarization-xsum": "staging/xsum_validation_100.jsonl"
}
```
Outputs: per-run CSVs and an aggregated `summary.csv` (avg latency/tokens/s/peak mem/acc/ROUGE). You can also provide task configs via `--task-configs` for custom datasets and metrics.

Alternatively, define explicit model/precision combos via `--model-configs`:
`model_configs.json`:
```json
[
  {"model": "mistralai/Mistral-7B-Instruct-v0.2", "bits": 16, "dtype": "float16"},
  {"model": "TheBloke/Llama-2-7B-Chat-GPTQ", "bits": null, "dtype": "float16"},
  {"model": "Qwen/Qwen1.5-7B-Chat", "bits": 4, "dtype": "float16", "use_chat_template": true}
]
```
Then run:
```bash
PYTHONPATH=. python scripts/run_matrix.py \
  --model-configs model_configs.json \
  --tasks classification-sst2 reasoning-gsm8k summarization-xsum \
  --data-map data_map.json \
  --out-dir staging/matrix_runs \
  --max-new-tokens 256 \
  --eval-rouge \
  --save-outputs
```
