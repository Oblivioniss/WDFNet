#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def custom_collate_fn(features):
    batch = {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "label": [f["label"] for f in features],
        "description": [f["description"] for f in features]
    }
    return batch

# Load config.json
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

df = pd.read_csv(config["csv_path"])
df["description"] = df["description"].astype(str)
df["image_path"] = df["image_path"].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])

filtered = df.groupby("label").filter(lambda x: len(x) > 3).reset_index(drop=True)

dataset = Dataset.from_dict({
    "description": filtered["description"].tolist(),
    "label": filtered["image_path"].tolist()
})

tokenizer = RobertaTokenizer.from_pretrained(config["model_path"])
model = RobertaForSequenceClassification.from_pretrained(config["model_path"])
model.eval().cuda()

def tokenize_fn(batch):
    return tokenizer(batch["description"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
dataloader = DataLoader(tokenized_dataset, batch_size=32, collate_fn=custom_collate_fn)

all_embeddings = []
all_labels = tokenized_dataset["label"]
all_texts = tokenized_dataset["description"]

for batch in tqdm(dataloader, desc="Encoding"):
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
    cls_vecs = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    all_embeddings.extend(cls_vecs.cpu().tolist())

with open(config["output_vec_path"], 'w', encoding='utf-8') as f:
    for label, text, vec in zip(all_labels, all_texts, all_embeddings):
        json.dump({
            "text": text,
            "label": label,
            "embedding": vec
        }, f)
        f.write('\n')

print(f"Saved embeddings to: {config["output_vec_path"]}")

vec_df = pd.DataFrame(all_embeddings)
meta_df = pd.DataFrame({
    "text": all_texts,
    "label": all_labels
})
csv_df = pd.concat([meta_df, vec_df], axis=1)
csv_df.to_csv("roberta_text_embeddings.csv", index=False)
print("Saved embeddings to roberta_text_embeddings.csv")