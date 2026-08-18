import json
import shutil
import kagglehub
import pandas as pd
from pathlib import Path


def asclepios_collect_and_organize_data(save_path):

    save_path = Path(save_path)

    data_path_1 = kagglehub.dataset_download("yousefsaeedian/ai-medical-chatbot")

    data_path_2  = kagglehub.dataset_download("divyanshu2000/doctor-healthcare-100k")

    data_path_1 = Path(data_path_1)

    data_path_2 = Path(data_path_2)

    asclepios_training_data = save_path / "asclepios_training_data"
    asclepios_training_data.mkdir(parents=True, exist_ok=True)

    for file in data_path_1.iterdir():
        shutil.move(str(file), str(asclepios_training_data))

    for file in data_path_2.iterdir():
        shutil.move(str(file), str(asclepios_training_data))

    csv_file_1 = asclepios_training_data / "ai-medical-chatbot.csv"

    csv_file_2 = asclepios_training_data / "Doctor-HealthCare-100k.csv"

    df_1 = pd.read_csv(csv_file_1)

    df_2 = pd.read_csv(csv_file_2)

    jsonl_file = asclepios_training_data / "asclepios_training_data.jsonl"

    with open(jsonl_file, "w", encoding="utf-8") as f:

        for _, row in df_1.iterrows():

            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": str(row["Patient"])
                    },
                    {
                        "role": "assistant",
                        "content": str(row["Doctor"])
                    }
                ]
            }

            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

        for _, row in df_2.iterrows():
        
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": str(row["input"])
                    },
                    {
                        "role": "assistant",
                        "content": str(row["output"])
                    }
                ]  
            }
        
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
        

    print(f"Created: {jsonl_file}")
    print(f"Training examples: {len(df_1) + len(df_2)}")


if __name__ == "__main__":
    asclepios_collect_and_organize_data(
        "/home/christianrafferty/Documents/projects/asclepios_ai/data"
    )