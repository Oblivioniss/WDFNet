#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FossilDataset(Dataset):
    def __init__(self, img_label_list, transform=None):
        self.img_label_list = img_label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, idx):
        img_path, label = self.img_label_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_split_list(split_file, original_dir, augmented_dir, class_to_idx):
    img_label_list = []
    with open(split_file, 'r') as f:
        for line in f:
            fname = line.strip()
            if 'augmented' in fname:
                img_path = os.path.join(augmented_dir, fname)
                base_class = fname.split('_augmented_')[0].split('_')[0]
            else:
                img_path = os.path.join(original_dir, fname)
                base_class = fname.split('_')[0]
            label = class_to_idx[base_class]
            img_label_list.append((img_path, label))
    return img_label_list

def prepare_datasets(original_dir, augmented_dir, split_dir, seed=42):
    random.seed(seed)

    class_to_images = defaultdict(list)
    for fname in os.listdir(original_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = fname.split('_')
        if len(parts) < 3:
            continue
        cls = parts[0]
        class_to_images[cls].append(fname)

    valid_classes = {k: v for k, v in class_to_images.items() if len(v) > 3}
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(valid_classes.keys()))}

    train_list = load_split_list(os.path.join(split_dir, 'train_list.txt'), original_dir, augmented_dir, class_to_idx)
    val_list = load_split_list(os.path.join(split_dir, 'val_list.txt'), original_dir, augmented_dir, class_to_idx)
    test_list = load_split_list(os.path.join(split_dir, 'test_list.txt'), original_dir, augmented_dir, class_to_idx)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return (
        FossilDataset(train_list, transform),
        FossilDataset(val_list, transform),
        FossilDataset(test_list, transform),
        class_to_idx
    )
