# Asclepios AI

Asclepios AI is a medical-information assistant with a Python web app, a model inference layer, and a training pipeline for a fine-tuned Qwen-based chatbot.

## What is in this project

- A lightweight frontend in `src/app/index.html`
- A Python backend in `src/app/server.py`
- Model logic in `src/model/asclepios_model.py`
- Training scripts in `src/model/asclepios_train.py`
- Data preparation utilities in `src/model/asclepios_data.py`

## Project layout

```text
asclepios_AI/
├── .venv/
├── .env.example
├── README.md
├── requirements.txt
├── data/
│   └── asclepios_training_data/
│       └── asclepios_training_data.jsonl
├── models/
│   ├── asclepios_model/
│   └── asclepios_lora/
├── src/
│   ├── app/
│   │   ├── index.html
│   │   ├── server.py
│   │   └── stylesheet.css
│   └── model/
│       ├── asclepios_data.py
│       ├── asclepios_model.py
│       ├── asclepios_pipeline.py
│       └── asclepios_train.py
└── docs/
    └── ASCLEPIOS_DOCS.md
```

## Reproducible naming conventions

Use these exact names for portability across machines:

- `asclepios_model` for the merged or loaded model directory
- `asclepios_lora` for the LoRA adapter directory
- `asclepios_training_data.jsonl` for the training data file

This avoids hardcoded absolute paths and makes the project easier to replicate on another machine.

## Setup

1. Create or activate the virtual environment

```bash
cd /path/to/asclepios_AI
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set optional environment variables

```bash
export ASCLEPIOS_MODEL_DIR="$PWD/models"
export ASCLEPIOS_MODEL_PATH="$PWD/models/asclepios_model"
export ASCLEPIOS_LORA_PATH="$PWD/models/asclepios_lora"
export ASCLEPIOS_DATA_DIR="$PWD/data"
export PORT=8000
export ASCLEPIOS_MODEL_PORT=8001
```

A sample file is included at `.env.example`.

## Run the app

```bash
cd /path/to/asclepios_AI
source .venv/bin/activate
node src/app/server.js
```

Open the app at the single public URL:

```text
http://localhost:8000
```

The Node server owns port `8000` and starts the Python model service on internal
port `8001`. Do not open the model-service port directly. Run `server.py` alone
only when debugging the Python service; it defaults to port `8000`.

## Notes

- The app is intentionally designed to work even when no local model checkpoint is present yet by falling back to a safe medical-information response.
- This is not medical diagnosis and should not replace professional clinical advice.
- For serious or urgent symptoms, users should seek urgent care or emergency services.

## Model details

The model is based on [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) and is intended for medical-information assistance using a fine-tuned training workflow.

## Data sources

The project references these public training datasets:

- AI Medical Chatbot — `yousefsaeedian/ai-medical-chatbot`
- Doctor-HealthCare-100k — `divyanshu2000/doctor-healthcare-100k`

## License

This project uses the Apache 2.0 license for the model data and training workflow described in the project. Review the dataset and model licenses before commercial or large-scale use.