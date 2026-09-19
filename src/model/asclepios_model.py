import json
import os
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

CRISIS_KEYWORDS = [
    "kill myself", "suicide", "suicidal", "want to die", "end my life",
    "end it all", "hurt myself", "self harm", "self-harm", "cant go on",
    "can't go on", "no reason to live", "better off dead",
]

CRISIS_RESPONSE = (
    "I'm really concerned about what you've shared. I'm not able to give "
    "medical advice for this — please reach out to a crisis line or emergency "
    "services right away. In the US you can call or text 988 (Suicide & "
    "Crisis Lifeline), or go to your nearest emergency room. If you're "
    "outside the US, please contact your local emergency number or a "
    "local crisis line. You don't have to go through this alone."
)

DEGENERATE_RESPONSE = (
    "I wasn't able to generate a reliable answer to that. Please consult a "
    "doctor, urgent care, or emergency services for medical concerns, "
    "especially if symptoms are severe or getting worse."
)


def contains_crisis_language(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def is_degenerate_repetition(text: str, min_repeats: int = 4, phrase_words: int = 3) -> bool:
    """Heuristic check for repetition-loop degeneration: looks for any
    short phrase (phrase_words long) that repeats min_repeats+ times."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < phrase_words * min_repeats:
        return False

    phrase_counts = {}
    for i in range(len(words) - phrase_words + 1):
        phrase = " ".join(words[i:i + phrase_words])
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        if phrase_counts[phrase] >= min_repeats:
            return True
    return False

def validate_tokenizer_config(checkpoint_path):
    """
    Validates tokenizer_config.json in the given checkpoint directory.
    Raises ValueError/FileNotFoundError with a clear reason if invalid.
    """
    tokenizer_config_path = checkpoint_path / "tokenizer_config.json"

    if not tokenizer_config_path.exists():
        raise FileNotFoundError(
            f"tokenizer_config.json not found in {checkpoint_path}"
        )

    if tokenizer_config_path.stat().st_size == 0:
        raise ValueError(
            f"tokenizer_config.json in {checkpoint_path} is empty (0 bytes)."
        )

    with open(tokenizer_config_path, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"tokenizer_config.json in {checkpoint_path} is corrupt/invalid JSON: {e}"
            ) from e

    if data is None:
        raise ValueError(
            f"tokenizer_config.json in {checkpoint_path} parsed to null/None."
        )


def is_raw_peft_adapter(path: Path) -> bool:
    """A raw (unmerged) PEFT/LoRA adapter dir has adapter_config.json but
    no full model weights (no config.json describing a base architecture)."""
    has_adapter_config = (path / "adapter_config.json").exists()
    has_full_config = (path / "config.json").exists()
    return has_adapter_config and not has_full_config

def load_model_and_tokenizer(path, base_model_name=None):
    """Shared loader. Handles two cases:
    1. A normal / merged model directory -> AutoModelForCausalLM.from_pretrained
    2. A raw PEFT adapter directory -> load base model, then apply adapter via PeftModel
    """
    path = Path(path)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if is_raw_peft_adapter(path):
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                f"{path} looks like a raw PEFT/LoRA adapter (has adapter_config.json, "
                "no config.json), but the `peft` package is not installed. "
                "Run `pip install peft` or merge the adapter into the base model first."
            )

        peft_config = PeftConfig.from_pretrained(path)
        resolved_base = base_model_name or peft_config.base_model_name_or_path
        if not resolved_base:
            raise RuntimeError(
                f"Could not determine base model for adapter at {path}. "
                "Pass base_model_name explicitly."
            )

        print(f"Detected raw PEFT adapter at {path}. Loading base model: {resolved_base}")
        tokenizer = AutoTokenizer.from_pretrained(resolved_base)
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, path)
        model.eval()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        device_map="auto",
    )
    return model, tokenizer


def get_model_and_tokenizer(model_save_path, base_model_name=None):
    """
    Looks for a model directly at model_save_path first (handles both
    merged models and raw PEFT adapters). Falls back to searching
    checkpoint-* subdirectories if that path doesn't exist or fails.

    NOTE: model_save_path should already point at the final model
    directory (e.g. .../asclepios_lora) -- we do NOT append
    "asclepios_lora" again, since that previously caused a duplicated
    path segment that could never exist.
    """
    local_model = Path(model_save_path)

    if local_model.exists() and local_model.is_dir():
        try:
            if is_raw_peft_adapter(local_model):

                return load_model_and_tokenizer(local_model, base_model_name)
            validate_tokenizer_config(local_model)
            return load_model_and_tokenizer(local_model, base_model_name)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            print(f"Model at {local_model} failed validation/load: {e}")

    checkpoint_dir = Path(model_save_path)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"No saved model or checkpoint directory found at {checkpoint_dir}. "
            "Run asclepios_train.py first."
        )

    checkpoints = [
        p for p in checkpoint_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}.")

    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]), reverse=True)

    last_error = None

    for i, candidate in enumerate(checkpoints):
        try:
            if is_raw_peft_adapter(candidate):
                model, tokenizer = load_model_and_tokenizer(candidate, base_model_name)
            else:
                validate_tokenizer_config(candidate)
                model, tokenizer = load_model_and_tokenizer(candidate, base_model_name)

            if i == 0:
                print(f"Using checkpoint: {candidate}")
            else:
                print(
                    f"Checkpoint {checkpoints[i - 1].name} failed to load — "
                    f"reverted to previous checkpoint: {candidate.name}"
                )

            return model, tokenizer

        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            print(f"Skipping {candidate.name} — {e}")
            last_error = e
            continue

    raise RuntimeError(
        "No usable checkpoint found — all checkpoints failed validation or loading."
    ) from last_error


def describe_model(model):
    """Print quick diagnostics so you can confirm whether a LoRA adapter
    is actually active on the loaded model."""
    is_peft = PEFT_AVAILABLE and isinstance(model, PeftModel)
    print(f"Model class: {type(model).__name__}")
    print(f"PEFT adapter active: {is_peft}")
    if is_peft:
        try:
            print(f"Active adapter(s): {model.active_adapters}")
        except Exception:
            pass


def load_model_with_fallback(primary_path, fallback_path, base_model_name=None):
    """Try the primary (LoRA) model path; on any expected failure, fall
    back to the base checkpoint directory. Errors are logged instead of
    silently swallowed."""
    try:
        return get_model_and_tokenizer(primary_path, base_model_name)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        print(f"Primary model load failed ({e}); falling back to {fallback_path}")
        return get_model_and_tokenizer(fallback_path, base_model_name)


SYSTEM_PROMPT = (
    "You are Asclepios, a practical medical-information assistant. Give a "
    "direct answer to the user's question first; do not begin with a generic "
    "disclaimer. For symptom questions, organize the response as: likely "
    "general possibilities (without diagnosing), what the user can do now, "
    "specific warning signs that need urgent care, and when to contact a "
    "clinician. Ask at most one focused follow-up question when key details "
    "are missing. Use plain language, concrete timeframes, and actionable "
    "examples. Never invent facts, medication doses, test results, or a "
    "diagnosis. You are not a doctor, so clearly say when professional care "
    "is needed."
)


def build_prompt(tokenizer, user_text, system_prompt=SYSTEM_PROMPT):
    """Applies the tokenizer's chat template so the model sees a proper
    system/user/assistant turn structure instead of raw text continuation."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_model_response(
    model,
    tokenizer,
    text,
    max_new_tokens=400,
    do_sample=False,
    temperature=0.7,
    system_prompt=SYSTEM_PROMPT,
):
    """Generate a response to `text`, returning only the newly generated
    tokens (not an echo of the prompt). Applies the chat template and
    guards against crisis-language input and degenerate/looping output."""

    if contains_crisis_language(text):
        return CRISIS_RESPONSE

    prompt = build_prompt(tokenizer, text, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if is_degenerate_repetition(response):
        return DEGENERATE_RESPONSE

    return response

def run_repl(model, tokenizer):
    print("Type 'quit' or 'exit' to stop, Ctrl+C also works.")
    while True:
        try:
            text = input("> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if text.strip().lower() in {"quit", "exit"}:
            break
        if not text.strip():
            continue

        try:
            response = get_model_response(model, tokenizer, text)
            print(response)
        except Exception as e:
            print(f"Generation failed: {e}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PRIMARY_PATH = Path(os.getenv("ASCLEPIOS_LORA_PATH", PROJECT_ROOT / "models" / "asclepios_lora"))
    FALLBACK_PATH = Path(os.getenv("ASCLEPIOS_MODEL_PATH", PROJECT_ROOT / "models" / "asclepios_model"))
    BASE_MODEL_NAME = os.getenv("ASCLEPIOS_BASE_MODEL_NAME")

    local_model, local_tokenizer = load_model_with_fallback(
        str(PRIMARY_PATH), str(FALLBACK_PATH), base_model_name=BASE_MODEL_NAME
    )
    describe_model(local_model)
    run_repl(local_model, local_tokenizer)