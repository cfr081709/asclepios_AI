from pathlib import Path
from asclepios_train import asclepios_train_model
from asclepios_data import asclepios_collect_and_organize_data
from asclepios_model import get_model_and_tokenizer, get_model_response

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG = {
    "epochs" : 1,
    "training_data_file_path": BASE_DIR / "data" / "asclepios_training_data" / "asclepios_training_data.jsonl",
    "model_save_path" : BASE_DIR / "models" / "asclepios_model",
    "save_path" : BASE_DIR / "data" / "asclepios_training_data" ,
}

if __name__ == "__main__":
    print("""
    Beginning asclepios_ai pipeline:
        
        PLEASE ESNURE CONFIGS ARE CORRECT FOR YOUR SYSTEM
        
        1. Data collection
        2. Model training
        3. Live model
""")

    begin = input("This may take a few hours - begin? (y/n) ")

    if begin == "y":
        asclepios_collect_and_organize_data(CONFIG["save_path"])
        asclepios_train_model(CONFIG["epochs"], CONFIG["training_data_file_path"], CONFIG["model_save_path"])
        model, tokenizer = get_model_and_tokenizer(CONFIG["model_save_path"])
        get_model_response(model, tokenizer)
    else:
        print("User entered 'n' script quitting...")
        exit()