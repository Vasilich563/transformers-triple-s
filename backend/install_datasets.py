# import kagglehub
#
# Download latest version
import kagglehub
from datasets import load_dataset
import json
import torch
from transformers import RobertaTokenizerFast

read_size = 1 * 1024 * 1024 * 1024 // 2

def on_file(filename, tokenizer, read_size, max_length, result_filename):

    with open(filename, 'r', encoding='utf-8') as fin:
        i = 1
        while True:
            print(i)
            print("Reading...")
            dataset = fin.read(read_size)

            if not dataset:
                break



        # train_tokens = tokenizer(
        #     train_dataset_text, truncation=True, padding="max_length", max_length=max_length, stride=0, return_tensors='np', return_overflowing_tokens=True
        # )
        #
        # train_dataset = [
        #     {"input_ids": train_tokens["input_ids"][i].tolist(), "hugging_face_mask": train_tokens["attention_mask"][i].tolist()}
        #     for i in range(train_tokens.input_ids.shape[0])
        # ]

            print("Tokenizing...")
            dataset = tokenizer(
                dataset, truncation=True, padding="max_length", max_length=max_length, stride=0, return_tensors='np', return_overflowing_tokens=True
            )

            print("Wrapping")
            dataset = [
                {"input_ids": dataset["input_ids"][i].tolist(), "hugging_face_mask": dataset["attention_mask"][i].tolist()}
                for i in range(dataset.input_ids.shape[0])
            ]

            # with open("train_dataset.json", 'w') as fout:
            #     json.dump(train_dataset, fout)
            print("Writing...")
            with open(f"{result_filename}part{i}.json", 'w') as fout:
                json.dump(dataset, fout)
            i += 1
            

def install_and_tokenize():
    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-large")
    max_length = 256

    #path = kagglehub.dataset_download("himonsarkar/openwebtext-dataset")

    #path = "C:/Users/amis-/PycharmProjects/semantic_search_system/backend/openwebtext-dataset"
    path = "C:/Users/amis-/PycharmProjects/semantic_search_system/backend/openwebtext-dataset-2"

    print("Path to dataset files:", path)
    # with open(f"{path}/train_split.txt", 'r', encoding='utf-8') as fin:
    #     train_dataset_text = fin.read()

    print("Train dataset")
    on_file(f"{path}/train_split.txt", tokenizer, read_size, max_length, "./new_datasets-2/train/")
    on_file(f"{path}/val_split.txt", tokenizer, read_size, max_length, "./new_datasets-2/val/")





def join_datasets():
    path = "C:/Users/amis-/PycharmProjects/semantic_search_system/backend/openwebtext-dataset"

    print("Path to dataset files:", path)
    with open(f"{path}/train_split.txt", 'a', encoding='utf-8') as fout:
        texts = [" " for i in range(1000000)]
        dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
        i = 1
        for item in dataset["train"]:
            texts[i] = item['text']
            i += 1
            if i == 1000000:
                i = 1
                fout.write(" ".join(texts))

        fout.write(" ".join(texts[:i]))

        # texts = [" " for _ in range(1000000 * 10)]
        # dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)
        # i = 1
        # for item in dataset["train"]:
        #     texts[i] = item["text"]
        #     i += 1
        #     if i == 1000000 * 10:
        #         i = 1
        #         fout.write(" ".join(texts))
        # fout.write(" ".join(texts[:i]))


def join_jsons(path, last_index):
    res = []
    for i in range(1, last_index + 1):
        print(f"Reading {i}...")
        with open(f"{path}/part{i}.json", 'r') as fin:
            data = json.load(fin)
            print("Joining...")
            res.extend(data)

    print("Writing...")
    with open(f"{path}/joined.json", 'w') as fout:
        json.dump(res, fout)


def join_jsons_to_torch(path, last_index, device, ids_dtype, mask_dtype):
    dataset = []
    for i in range(1, last_index + 1):
        print(f"Reading {i}...")
        with open(f"{path}/part{i}.json", 'r') as fin:
            data = json.load(fin)

        print(len(data))
        print("Converting to torch...")
        for j in range(len(data)):
            data[j]["input_ids"] = torch.tensor(data[j]["input_ids"], dtype=ids_dtype, device=device, requires_grad=False)
            data[j]["hugging_face_mask"] = torch.tensor(data[j]["hugging_face_mask"], dtype=mask_dtype, device=device, requires_grad=False)

        print("Joining...")
        dataset.extend(data)

    print(f"Done, {len(dataset)} rows")

    from torch.utils.data import DataLoader
    from transformers.data.data_collator import DataCollatorForLanguageModeling
    from transformers import RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-large")
    mlm_probability = 0.15
    mlm = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mlm_probability, return_tensors='pt')
    loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=mlm)
    for i, x in enumerate(loader):
        print(i, x)
        if i == 3:
            print(tokenizer.decode(x["input_ids"]))
            break


    # dataset = [
    #     {"input_ids": tokens["input_ids"][i], "hugging_face_mask": tokens["attention_mask"][i]} for i in range(tokens.input_ids.shape[0])
    # ]




#join_datasets()
#install_and_tokenize()

# TODO 70 files
#join_jsons_to_torch("C:/Users/amis-/PycharmProjects/semantic_search_system/backend/new_datasets/train", 52, torch.device("cpu"), torch.uint16, torch.int8)
join_jsons_to_torch("C:/Users/amis-/PycharmProjects/semantic_search_system/backend/new_datasets/train", 1, torch.device("cpu"), torch.uint16, torch.int8)


# https://www.kaggle.com/datasets/himonsarkar/openwebtext-dataset?select=train_split.txt
# https://huggingface.co/datasets/bookcorpus/bookcorpus
# https://huggingface.co/datasets/legacy-datasets/wikipedia
# TODO Path to dataset files: /home/yackub/.cache/kagglehub/datasets/himonsarkar/openwebtext-dataset/versions/1