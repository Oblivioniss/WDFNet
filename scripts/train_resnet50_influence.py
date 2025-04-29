#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from data_processor.data_preprocess_resnet50 import prepare_datasets
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from typing import Any, Dict, List, Optional, Union

# Load config.json
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Configurations
os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
data_dir = config["data_dir"]
augmented_dir = config["augmented_dir"]
num_epochs = config["resnet_num_epochs"]
batch_size = config["resnet_batch_size"]
learning_rate = config["resnet_lr"]
weight_decay = config["resnet_weight_decay"]
step_size = config["resnet_step_size"]
gamma = config["resnet_gamma"]
split_save_dir = config["split_save_dir"]

# Load datasets
train_set, val_set, test_set, class_to_idx = prepare_datasets(data_dir, augmented_dir, split_save_dir)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
num_classes = len(class_to_idx)

# Load ResNet-50
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Training
for epoch in range(num_epochs):
    model.train()
    running_loss = correct = total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    scheduler.step()
    train_acc = correct / total
    avg_loss = running_loss / total
    print(f"Epoch [{epoch+1:02d}] Done | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}", end='')

    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
        val_acc = evaluate(model, val_loader)
        print(f" | Val Acc: {val_acc:.4f}")
    else:
        print()

test_acc = evaluate(model, test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Kronfluence analysis
print("\nPreparing Kronfluence influence computation...")

class FossilTask(Task):
    def compute_train_loss(self, batch: Any, model: nn.Module, sample: bool = False) -> torch.Tensor:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        return nn.CrossEntropyLoss()(outputs, labels)

    def compute_measurement(self, batch: Any, model: nn.Module) -> torch.Tensor:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        return nn.CrossEntropyLoss()(outputs, labels)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        return None

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        return None

task = FossilTask()
model = prepare_model(model, task)

analyzer = Analyzer(
    analysis_name="fossil_resnet50",
    model=model,
    task=task,
)

analyzer.fit_all_factors(
    factors_name="resnet50_factors",
    dataset=train_set,
    per_device_batch_size=batch_size,
)

analyzer.compute_pairwise_scores(
    scores_name="resnet50_scores",
    factors_name="resnet50_factors",
    query_dataset=val_set,
    train_dataset=train_set,
    per_device_query_batch_size=batch_size,
)
