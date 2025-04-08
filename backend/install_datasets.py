# import kagglehub
#
# Download latest version
import kagglehub
from datasets import load_dataset
import json
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-large")
max_length = 256

path = kagglehub.dataset_download("himonsarkar/openwebtext-dataset")

print("Path to dataset files:", path)
with open(path, 'r') as fin:
    train_dataset_text = fin.read("train_split.txt")
    val_dataset_text = fin.read("val_split.txt")


wiki_dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
bookcorpus_dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)

for wiki_item in wiki_dataset["train"]:
    train_dataset_text += f" {wiki_item['text']} "

for bookcorpus_item in bookcorpus_dataset["train"]:
    train_dataset_text += f" {bookcorpus_item["text"]} "

train_tokens = tokenizer(
    train_dataset_text, truncation=True, padding="max_length", max_length=max_length, stride=0, return_tensors='np', return_overflowing_tokens=True
)

train_dataset = [
    {"input_ids": train_tokens["input_ids"][i].tolist(), "hugging_face_mask": train_tokens["attention_mask"][i].tolist()}
    for i in range(train_tokens.input_ids.shape[0])
]

val_tokens = tokenizer(
    val_dataset_text, truncation=True, padding="max_length", max_length=max_length, stride=0, return_tensors='np', return_overflowing_tokens=True
)

val_dataset = [
    {"input_ids": val_tokens["input_ids"][i].tolist(), "hugging_face_mask": val_tokens["attention_mask"][i].tolist()}
    for i in range(val_tokens.input_ids.shape[0])
]

with open("dataset.json", 'w') as fout:
    json.dump(
        {
            "train": train_dataset, "val": val_dataset
        },
        fout
    )

# https://www.kaggle.com/datasets/himonsarkar/openwebtext-dataset?select=train_split.txt
# https://huggingface.co/datasets/bookcorpus/bookcorpus
# https://huggingface.co/datasets/legacy-datasets/wikipedia
# TODO Path to dataset files: /home/yackub/.cache/kagglehub/datasets/himonsarkar/openwebtext-dataset/versions/1