#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import json
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

class FixedMorphodynamicAugmentor:
    def __init__(self):
        pass

    def augment_affine(self, img: np.ndarray) -> np.ndarray:
        rows, cols = img.shape
        dx = cols * random.uniform(0.02, 0.08)
        dy = rows * random.uniform(0.02, 0.08)
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([
            [dx, dy],
            [cols - dx - 1, dy],
            [dx, rows - dy - 1]
        ])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(img, M, (cols, rows), borderValue=255)

    def augment_morph(self, img: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        if random.random() < 0.5:
            return cv2.erode(img, kernel, iterations=random.randint(1, 2))
        else:
            return cv2.dilate(img, kernel, iterations=random.randint(1, 2))

    def augment_noise(self, img: np.ndarray) -> np.ndarray:
        vals, counts = np.unique(img, return_counts=True)
        background_val = int(vals[np.argmax(counts)])
        fossil_mask = (img < background_val - 10).astype(np.uint8)
        fossil_mask = cv2.dilate(fossil_mask, np.ones((5, 5), np.uint8), iterations=1)
        noise = np.random.normal(loc=0, scale=10, size=img.shape).astype(np.int16)
        img_int = img.astype(np.int16)
        mask = (fossil_mask == 1)
        img_int[mask] += noise[mask]
        img_int = np.clip(img_int, 0, 255).astype(np.uint8)
        return img_int

    def augment_broken(self, img: np.ndarray, crack_width_range=(10, 25)) -> np.ndarray:
        h, w = img.shape
        vals, counts = np.unique(img, return_counts=True)
        background_val = int(vals[np.argmax(counts)])
        fossil_mask = (img < background_val - 10).astype(np.uint8)
        fossil_mask = cv2.dilate(fossil_mask, np.ones((5, 5), np.uint8), iterations=1)

        crack_mask = np.ones((h, w), dtype=np.uint8)
        sides = ['left', 'right', 'top', 'bottom']
        side = random.choice(sides)
        if side in ['left', 'right']:
            x = 0 if side == 'left' else w - 1
            y = np.random.randint(0, h)
            end_x = w - 1 if side == 'left' else 0
            end_y = np.random.randint(0, h)
        else:
            y = 0 if side == 'top' else h - 1
            x = np.random.randint(0, w)
            end_y = h - 1 if side == 'top' else 0
            end_x = np.random.randint(0, w)

        num_points = np.random.randint(6, 10)
        xs = np.linspace(x, end_x, num_points)
        ys = np.linspace(y, end_y, num_points)
        points = []
        for xi, yi in zip(xs, ys):
            jitter_x = int(np.clip(xi + np.random.uniform(-50, 50), 0, w - 1))
            jitter_y = int(np.clip(yi + np.random.uniform(-50, 50), 0, h - 1))
            points.append((jitter_x, jitter_y))
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        thickness = np.random.randint(*crack_width_range)
        cv2.polylines(crack_mask, [pts], isClosed=False, color=0, thickness=thickness)

        result = img.copy()
        crack_area = (crack_mask == 0)
        result[crack_area] = background_val

        fossil_crack_area = np.logical_and(crack_area, fossil_mask == 1)
        noise = np.random.normal(loc=0, scale=30, size=img.shape).astype(np.int16)
        result = result.astype(np.int16)
        result[fossil_crack_area] += noise[fossil_crack_area]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def augment_contrast(self, img: np.ndarray) -> np.ndarray:
        vals, counts = np.unique(img, return_counts=True)
        background_val = int(vals[np.argmax(counts)])
        img = img.astype(np.int16)
        contrast = img.copy()

        alpha_dark = 1.15
        alpha_bright = 1.15

        brighter_mask = (img > background_val + 5)
        contrast[brighter_mask] = img[brighter_mask] * alpha_bright

        darker_mask = (img < background_val - 5)
        contrast[darker_mask] = img[darker_mask] / alpha_dark

        background_mask = (np.abs(img - background_val) <= 5)
        contrast[background_mask] = background_val

        contrast = np.clip(contrast, 0, 255).astype(np.uint8)
        return contrast

    def generate_all(self, image: np.ndarray) -> list:
        assert image.ndim == 2
        image = image.copy()
        return [
            self.augment_affine(image),
            self.augment_morph(image),
            self.augment_noise(image),
            self.augment_broken(image),
            self.augment_contrast(image)
        ]

augmentor = FixedMorphodynamicAugmentor()

def visualize_fixed_augmentation(original_path, save_path=None):
    img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image:", original_path)
        return

    augmentor = FixedMorphodynamicAugmentor()
    affine_img, morph_img, noise_img, broken_img, contrast_img = augmentor.generate_all(img)

    fig, axs = plt.subplots(2, 3, figsize=(8, 8))
    axs = axs.flatten()
    titles = ["Original", "Affine", "Morph", "Noise", "Broken", "Contrast"]
    images = [img, affine_img, morph_img, noise_img, broken_img, contrast_img]

    for i in range(6):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def apply_random_combinations(image, all_augment_fns, n=3):
    all_pairs = list(itertools.combinations(range(len(all_augment_fns)), 2))
    random.shuffle(all_pairs)

    selected_combos = []
    used_indices = set()

    for combo in all_pairs:
        if len(selected_combos) == n:
            break
        temp_used = used_indices | set(combo)
        if len(temp_used) <= len(all_augment_fns):
            selected_combos.append(combo)
            used_indices = temp_used

    unused = set(range(len(all_augment_fns))) - used_indices
    if unused and len(selected_combos) < n:
        for idx in unused:
            for j in range(len(all_augment_fns)):
                if idx != j and (idx, j) not in selected_combos and (j, idx) not in selected_combos:
                    selected_combos.append((idx, j))
                    if len(selected_combos) == n:
                        break
            if len(selected_combos) == n:
                break

    results = []
    for i, (a, b) in enumerate(selected_combos):
        aug_img = all_augment_fns[a](image.copy())
        aug_img = all_augment_fns[b](aug_img)
        results.append((i + 1, aug_img))
    return results

def generate_augmented_dataset(augmentor, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    augment_fns = [
        augmentor.augment_affine,
        augmentor.augment_morph,
        augmentor.augment_noise,
        augmentor.augment_broken,
        augmentor.augment_contrast
    ]

    for fname in tqdm(image_list, desc="Generating augmented images"):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        base_name = os.path.splitext(fname)[0]
        augmented_images = apply_random_combinations(img, augment_fns, n=3)

        for idx, aug_img in augmented_images:
            save_name = f"{base_name}_augmented_{idx}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, aug_img)

config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

generate_augmented_dataset(
    augmentor,
    input_dir=config["data_dir"],
    output_dir=["augmented_dir"]
)
