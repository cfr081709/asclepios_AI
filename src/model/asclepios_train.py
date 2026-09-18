import torch
import platform
from trl import (
    SFTTrainer,
    SFTConfig
)
from peft import (
    LoraConfig,
    get_peft_model
)
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from datasets import load_dataset

def asclepios_train_model(epochs, training_data_file_path, model_save_path):

    MODEL = "Qwen/Qwen2.5-3B-Instruct"

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        torch.device("mps")
    elif torch.cuda.is_available():
        torch.device("cuda")
    else:
        raise Exception("Cuda/MPS unavailable - ending training")

    dataset = load_dataset(
        "json",
        data_files=training_data_file_path
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config = bnb_config,
        device_map = "auto"
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=f"{model_save_path}/asclepios_model/checkpoint_saves",

        num_train_epochs=epochs,

        per_device_train_batch_size=2,

        gradient_accumulation_steps=8,

        learning_rate=2e-4,

        logging_steps=10,

        save_steps=100,

        fp16=False,

        bf16=False,

        max_length=512
    )

    dataset = dataset["train"].shuffle(seed=42)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer
    )

    trainer.train()

    model.save_pretrained(f"{model_save_path}/asclepios_lora")
    tokenizer.save_pretrained(f"{model_save_path}/asclepios_lora")

if __name__ == "__main__":
    asclepios_train_model(1, "/home/christianrafferty/Documents/projects/asclepios_ai/data/asclepios_training_data/asclepios_training_data.jsonl", "/home/christianrafferty/Documents/projects/asclepios_ai/models")