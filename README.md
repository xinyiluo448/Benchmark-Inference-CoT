## Benchmark-Inference-CoT

Quick start for SST-2 (dev split):
```bash
# prepare jsonl from HF SST-2 dev
python3 data_prep/prepare_sst2.py --split validation --out sst2_dev.jsonl

# run benchmark on the converted file
python bench_framework.py --task classification-sst2 --data sst2_dev.jsonl --save-csv sst2_run.csv
```

Lightweight LLM inference benchmarking scaffold. Loads HuggingFace models (default `mistralai/Mistral-7B-Instruct-v0.2`), runs built-in mini tasks or custom datasets, reports latency, tokens/s, and peak GPU memory, and can save per-sample CSV.

### Layout
- `bench_framework.py`: CLI entry, argument parsing and orchestration.
- `bench_runner/`:
  - `model.py`: model/tokenizer loading (optional bitsandbytes 4-bit).
  - `tasks.py`: built-in tasks plus dataset loading/sampling helpers.
  - `runner.py`: single-sample inference, timing, and CSV writing.
- `data_prep/prepare_sst2.py`: convert HF `glue/sst2` into jsonl.
- `data_prep/prepare_gsm8k.py`: convert HF `gsm8k` into jsonl.
- `data_prep/prepare_xsum.py`: convert HF `xsum` into jsonl.
- `scripts/`: convenience entrypoints (`run_sst2_dev.sh`, `run_gsm8k.sh`, `run_xsum.sh`, `run_custom.sh`).
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

### Key arguments
- `--model`: HF model id (default `mistralai/Mistral-7B-Instruct-v0.2`)
- `--task`: built-in task key (`classification-sst2`, `reasoning-gsm8k`, `summarization-xsum`); can be overridden by `--data`
- `--data`: path to jsonl/json/csv
- `--max-samples` / `--shuffle-seed`: sampling control
- `--max-new-tokens`: generation limit
- `--load-4bit`: enable bitsandbytes 4-bit (requires installation)
- `--save-csv`: path to save per-sample results
- `--save-outputs`: also persist prompt/output/reference in CSV (useful for offline metrics like ROUGE)
- `--eval-rouge`: compute ROUGE-1/2/L in-process (requires `rouge-score`; needs references and outputs)

### Notes
- Requires a working python/python3 env with `transformers`, `torch`, `datasets` (for data prep), and bitsandbytes if using 4-bit.
- End-to-end time depends on GPU, tokens/s, and sample count; try small runs via `--max-samples` first.
- CSV outputs now save metrics only by default (name, latency_s, tokens_generated, tokens_per_s, peak_mem_gb, reference, correct); prompts/outputs can be stored via `--save-outputs`. The CLI prints per-sample details and a summary (avg latency, throughput, peak mem, accuracy with numeric extraction for GSM8K or exact match on `reference`). ROUGE F1 can be computed inline via `--eval-rouge` if `rouge-score` is installed.
