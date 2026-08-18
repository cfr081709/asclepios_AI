# Asclepios

Asclepios is a pipeline for fine-tuning a small open-weight LLM (Qwen2.5-3B-Instruct)
into a medical-Q&A chatbot using LoRA. It has four parts:

| File | Responsibility |
|---|---|
| `asclepios_data.py` | Download and merge public medical Q&A datasets into one JSONL training file |
| `asclepios_train.py` | LoRA fine-tune Qwen2.5-3B-Instruct on that JSONL file |
| `asclepios_model.py` | Load the trained model/tokenizer and run inference |
| `asclepios_pipeline.py` | Orchestrate the three steps above end-to-end |

---

## 1. `asclepios_data.py`

### `asclepios_collect_and_organize_data(save_path)`

Downloads two Kaggle datasets via `kagglehub`, merges them into a single
OpenAI-style chat-format JSONL file, and writes it to disk.

**Parameters**
- `save_path` (str | Path) — directory under which a `asclepios_training_data/`
  subfolder is created.

**What it does**
1. Downloads `yousefsaeedian/ai-medical-chatbot` and `divyanshu2000/doctor-healthcare-100k`
   from Kaggle Hub.
2. Moves all files from both dataset caches into `<save_path>/asclepios_training_data/`.
3. Reads `ai-medical-chatbot.csv` (columns: `Patient`, `Doctor`) and
   `Doctor-HealthCare-100k.csv` (columns: `input`, `output`).
4. Converts every row from both files into the chat schema:
   ```json
   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```
5. Writes all records, one JSON object per line, to
   `asclepios_training_data/asclepios_training_data.jsonl`.
6. Prints the output path and total example count.

**Requirements**: `kagglehub` (and valid Kaggle API credentials), `pandas`.

**Notes / gotchas**
- Both source CSVs are consumed entirely into memory via `pandas.read_csv`.
- Column names (`Patient`/`Doctor`, `input`/`output`) are hardcoded — if the
  upstream Kaggle datasets change their schema, this will raise a `KeyError`.
- The `__main__` block hardcodes a local path
  (`/home/christianrafferty/Documents/projects/asclepios_ai/data`) — update this
  before running standalone, or invoke the function directly with your own path.

---

## 2. `asclepios_train.py`

### `asclepios_train_model(epochs, training_data_file_path, model_save_path)`

Fine-tunes `Qwen/Qwen2.5-3B-Instruct` with 4-bit quantization + LoRA using
`trl.SFTTrainer`, then saves the resulting adapter and tokenizer.

**Parameters**
- `epochs` (int) — number of training epochs.
- `training_data_file_path` (str | Path) — path to the JSONL file produced by
  `asclepios_data.py`.
- `model_save_path` (str | Path) — base directory for checkpoints and the final
  LoRA adapter.

**What it does**
1. Selects a device (`mps` on Apple Silicon, `cuda` if available); raises an
   `Exception` if neither is available (no CPU fallback).
2. Loads the JSONL dataset via `datasets.load_dataset("json", ...)`.
3. Loads the base model in 4-bit (`BitsAndBytesConfig`, `fp16` compute dtype).
4. Wraps it with a LoRA adapter (`r=32`, `alpha=32`, dropout `0.05`, targeting
   `q_proj`, `k_proj`, `v_proj`, `o_proj`, task type `CAUSAL_LM`).
5. Shuffles the dataset (seed `42`) and trains with `SFTTrainer` using:
   - `output_dir = {model_save_path}/asclepios_model/checkpoint_saves`
   - batch size 2, gradient accumulation 8, learning rate `2e-4`
   - logging every 10 steps, checkpoint every 100 steps
   - `fp16=False`, `bf16=False`, `max_length=512`
6. Saves the LoRA adapter and tokenizer to `{model_save_path}/asclepios_lora`.

**Requirements**: `torch`, `trl`, `peft`, `transformers`, `datasets`, `bitsandbytes`,
a CUDA GPU or Apple Silicon (MPS) — no CPU-only training path.

**Notes / gotchas**
- `torch.device("mps")` / `torch.device("cuda")` are called but their return
  value is discarded — device placement for training actually relies on
  `device_map="auto"` in `from_pretrained`, not on these calls.
- 4-bit quantization + `fp16=False`/`bf16=False` in `SFTConfig` means training
  compute precision is effectively driven by `bnb_4bit_compute_dtype=torch.float16`.
- `model_save_path` gets `/asclepios_model/checkpoint_saves` appended for
  checkpoints but the final adapter is saved directly under
  `{model_save_path}/asclepios_lora` (one directory level shallower) — worth
  double-checking these line up with what `asclepios_model.py` expects.
- The `__main__` block again hardcodes a local absolute path.

---

## 3. `asclepios_model.py`

Loads a trained model (LoRA adapter or raw checkpoint) and runs interactive
inference.

### `validate_tokenizer_config(checkpoint_path)`
Sanity-checks a candidate model directory before loading:
- Raises `FileNotFoundError` if `tokenizer_config.json` is missing.
- Raises `ValueError` if it's empty, invalid JSON, or parses to `null`.

### `load_model_and_tokenizer(path)`
Shared loader: loads tokenizer + causal LM from `path`, using `float16` on
CUDA or `float32` otherwise, with `device_map="auto"`.

### `get_model_and_tokenizer(model_save_path)`
Resolves which weights to load, in priority order:
1. **LoRA adapter** — if `{model_save_path}/asclepios_lora` exists, validate and
   load it directly.
2. **Checkpoints** — otherwise, look inside `model_save_path` for directories
   named `checkpoint-<N>`, sort newest-first by the numeric suffix, and try
   each in turn until one passes `validate_tokenizer_config` and loads
   successfully. Prints which checkpoint was used (and if it had to fall back
   from a newer, broken one).
3. Raises `FileNotFoundError` if `model_save_path` doesn't exist or no
   `checkpoint-*` directories are found, or `RuntimeError` if every checkpoint
   fails.

### `get_model_response(model, tokenizer)`
Prompts the user via `input("> ")`, tokenizes, runs greedy generation
(`do_sample=False`, `max_new_tokens=100`, `repetition_penalty=1.1`), and
returns the decoded string (special tokens stripped).

**`__main__` behavior**: loads the model from a hardcoded path, then loops
`get_model_response` forever (one prompt/response per iteration, no exit
condition other than `Ctrl+C`).

---

## 4. `asclepios_pipeline.py`

End-to-end orchestrator that chains data collection → training → interactive
inference behind a single `y/n` confirmation prompt.

**Config** (`CONFIG` dict, paths relative to `BASE_DIR = parents[2]` of this
file — i.e. two directories up from wherever `asclepios_pipeline.py` lives):

| Key | Value |
|---|---|
| `epochs` | `1` |
| `training_data_file_path` | `<BASE_DIR>/data/asclepios_training_data/asclepios_training_data.jsonl` |
| `model_save_path` | `<BASE_DIR>/models/asclepios_model` |
| `save_path` | `<BASE_DIR>/data/asclepios_training_data` |

**Flow**: prompts "This may take a few hours - begin? (y/n)"; if `y`, runs
`asclepios_collect_and_organize_data` → `asclepios_train_model` →
`get_model_and_tokenizer` → `get_model_response` (once).

---

## Suggested run order (manual)

```bash
python asclepios_data.py       # produces asclepios_training_data.jsonl
python asclepios_train.py      # produces the asclepios_lora adapter
python asclepios_model.py      # interactive Q&A loop
```

## Dependencies (union across all files)

```
kagglehub
pandas
torch
transformers
trl
peft
datasets
bitsandbytes
```

## Data & disclaimer

Training data comes from two public Kaggle datasets of patient/doctor Q&A
exchanges. This is a research/prototype project — the resulting model is
**not a medical device** and outputs should not be treated as clinical advice.

## Dataset Licensing

Both source datasets are licensed under the **Apache License, Version 2.0**.

- **AI Medical Chatbot** — `yousefsaeedian/ai-medical-chatbot` (Kaggle)
  https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot
- **Doctor-HealthCare-100k** — `divyanshu2000/doctor-healthcare-100k` (Kaggle)
  https://www.kaggle.com/datasets/divyanshu2000/doctor-healthcare-100k

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```