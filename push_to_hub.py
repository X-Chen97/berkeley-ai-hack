import json
import pandas as pd
from datasets import load_dataset
from huggingface_hub import notebook_login

#notebook_login()

def save_as_jsonl(data, name):
    df = pd.DataFrame.from_records(data)
    df.to_json(f"{name}.jsonl", orient="records", lines=True)
    return None

with open("finetune_annotations.json") as f:
	data_f = f.read()
data = json.loads(data_f)

train = data["train"]
test = data["test"]
val = data["validation"]

print(len(train), len(test), len(val))


save_as_jsonl(train, "train")
save_as_jsonl(test, "test")
save_as_jsonl(val, "val")


data_files = {"train":"train.jsonl", "test":"test.jsonl", "validation":"val.jsonl"}

dataset_com = load_dataset("json", data_files=data_files)
dataset_com.push_to_hub("MaterialsAI/robocr_poscar", private=False)