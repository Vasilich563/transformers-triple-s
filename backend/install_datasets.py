# import kagglehub
#
# Download latest version
import kagglehub
path = kagglehub.dataset_download("himonsarkar/openwebtext-dataset")

print("Path to dataset files:", path)

from datasets import load_dataset

wiki_dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
bookcorpus_dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)

# https://www.kaggle.com/datasets/himonsarkar/openwebtext-dataset?select=train_split.txt
# https://huggingface.co/datasets/bookcorpus/bookcorpus
# https://huggingface.co/datasets/legacy-datasets/wikipedia
# TODO Path to dataset files: /home/yackub/.cache/kagglehub/datasets/himonsarkar/openwebtext-dataset/versions/1