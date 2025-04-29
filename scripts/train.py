#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from models.paleocellnet import PaleoCellNet, QuasiBiologicalPreprocessor
from data_processor.data_preprocess import FossilDataset, prepare_datasets

# Load config
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    images, labels, names = zip(*batch)
    return list(images), torch.tensor(labels), list(names)

preprocessor = QuasiBiologicalPreprocessor(
    patch_size=config["patch_size"],
    num_patches=config["num_patches"],
    add_positional_encoding=True
)
to_tensor = transforms.ToTensor()

def load_split_list(filename):
    with open(os.path.join(config["split_save_dir"], filename), 'r') as f:
        return [line.strip() for line in f]

train_files = load_split_list("train_list.txt")
val_files = load_split_list("val_list.txt")
test_files = load_split_list("test_list.txt")

train_set, val_set, test_set, class_to_idx = prepare_datasets(
    original_dir=config["data_dir"],
    augmented_dir=config["augmented_dir"],
    split_dir=config["split_save_dir"],
)

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=custom_collate_fn)
num_classes = len(class_to_idx)

# Load text embeddings
with open(config["embedding_json"], 'r', encoding='utf-8') as f:
    embedding_dict = {json.loads(line)["label"]: torch.tensor(json.loads(line)["embedding"], dtype=torch.float32) for line in f}

# Load influence scores
influence_data = load_file(config["influence_path"])
scores = influence_data["all_modules"].cpu().numpy()
avg_influence = scores.mean(axis=0)
threshold = np.percentile(np.abs(avg_influence), 95)
avg_influence = np.clip(avg_influence, -threshold, threshold)
min_val, max_val = avg_influence.min(), avg_influence.max()
avg_influence = (avg_influence - min_val) / (max_val - min_val + 1e-8)
avg_influence = 0.5 + 0.5 * avg_influence
avg_influence_tensor = torch.tensor(avg_influence, dtype=torch.float32).to(device)

# Model and losses
model = PaleoCellNet(num_classes=num_classes, use_pos_embedding=True).to(device)
criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
criterion_cos = nn.CosineEmbeddingLoss(margin=0.1)

optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs_pil, labels, names in tqdm(dataloader, desc="Evaluating", unit="batch"):
            labels = labels.to(device)
            imgs_tensor_raw = [torch.tensor(np.array(img), dtype=torch.uint8).unsqueeze(0) for img in imgs_pil]
            x_patches = [preprocessor.pad_and_partition(img) for img in imgs_tensor_raw]
            x_patch = torch.stack(x_patches).to(device)
            x_imgs_resized = [transforms.Resize((config["resize_size"], config["resize_size"]))(img) for img in imgs_pil]
            x_img_tensor = torch.stack([to_tensor(img) for img in x_imgs_resized]).to(device)
            outputs, _, _ = model(x_img_tensor, x_patch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Training
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = correct = total = 0
    for batch_idx, (imgs_pil, labels, names) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")):
        labels = labels.to(device)
        imgs_tensor_raw = [torch.tensor(np.array(img), dtype=torch.uint8).unsqueeze(0) for img in imgs_pil]
        x_patches = [preprocessor.pad_and_partition(img) for img in imgs_tensor_raw]
        x_patch = torch.stack(x_patches).to(device)
        x_imgs_resized = [transforms.Resize((config["resize_size"], config["resize_size"]))(img) for img in imgs_pil]
        x_img_tensor = torch.stack([to_tensor(img) for img in x_imgs_resized]).to(device)

        text_embeddings = []
        for name in names:
            base = name.split("_augmented_")[0] if "augmented" in name else name
            base = os.path.splitext(base)[0]
            text_embeddings.append(embedding_dict.get(base, torch.zeros(768)))
        text_embeddings = torch.stack(text_embeddings).to(device)

        cls_main, cls_cbm, cls_aux = model(x_img_tensor, x_patch)
        proj_cbm = F.normalize(cls_cbm[:, :768], dim=-1)
        proj_text = F.normalize(text_embeddings, dim=-1)
        loss_cos = criterion_cos(proj_cbm, proj_text, torch.ones_like(labels, dtype=torch.float32).to(device))

        batch_start_idx = batch_idx * config["batch_size"]
        batch_indices = torch.arange(batch_start_idx, batch_start_idx + labels.size(0)).to(device)
        influence_weights = avg_influence_tensor[batch_indices]

        loss_cls = criterion_cls(cls_main, labels)
        loss_cls = (loss_cls * influence_weights).mean()
        loss_aux = criterion_cls(cls_aux, labels)
        loss_aux = (loss_aux * influence_weights).mean()

        loss = loss_cls + config["aux_weight"] * loss_aux + config["cos_weight"] * loss_cos

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_img_tensor.size(0)
        _, preds = torch.max(cls_main, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_acc = correct / total
    print(f"Epoch {epoch+1:02d} | Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.4f}")

    if (epoch + 1) % 1 == 0 or (epoch + 1) == config["num_epochs"]:
        val_acc = evaluate(model, val_loader)
        print(f" | Val Acc: {val_acc:.4f}")
        test_acc = evaluate(model, test_loader)
        print(f" | Test Acc: {test_acc:.4f}")

    if (epoch + 1) % config["save_checkpoint_every"] == 0:
        checkpoint_path = f"checkpoints/paleocellnet_epoch{epoch+1}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

# Save final model
torch.save(model.state_dict(), f"paleocellnet_final.pth")
print("[*] Final model saved.")
