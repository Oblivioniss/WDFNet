#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import cv2
import os

from wtconv import WTConv2d
from torchvision.models import resnet50,ResNet50_Weights


class QuasiBiologicalPreprocessor:
    def __init__(self, patch_size=128, num_patches=8, add_positional_encoding=False):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.add_positional_encoding = add_positional_encoding

    def pad_and_partition(self, img_tensor):
        img = img_tensor.squeeze().cpu().numpy()
        h, w = img.shape

        th = int(np.ceil(h / self.patch_size) * self.patch_size)
        tw = int(np.ceil(w / self.patch_size) * self.patch_size)
        pad_h = (th - h) // 2
        pad_w = (tw - w) // 2
        img = np.pad(img, ((pad_h, th - h - pad_h), (pad_w, tw - w - pad_w)), mode='constant', constant_values=255)

        if self.add_positional_encoding:
            H, W = img.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid_x = grid_x.float() / (W - 1)
            grid_y = grid_y.float() / (H - 1)
            pos_x = grid_x.unsqueeze(0).numpy()
            pos_y = grid_y.unsqueeze(0).numpy()
            img_tensor = torch.tensor(np.stack([img, pos_x[0] * 255, pos_y[0] * 255], axis=0), dtype=torch.float32)
        else:
            img_tensor = torch.tensor(img).unsqueeze(0).float()

        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)

        bg_val = np.bincount(img.flatten().astype(np.uint8)).argmax()
        fossil_mask = (img < bg_val - 10).astype(np.uint8)
        fossil_mask = cv2.dilate(fossil_mask, np.ones((5, 5), np.uint8), iterations=1)

        ps = self.patch_size
        coords = []
        grad_scores = []
        for y in range(0, th - ps + 1, ps // 2):
            for x in range(0, tw - ps + 1, ps // 2):
                patch_mask = fossil_mask[y:y+ps, x:x+ps]
                if patch_mask.mean() < 0.1:
                    continue
                patch_grad = grad[y:y+ps, x:x+ps]
                grad_scores.append(patch_grad.sum())
                coords.append((x, y))

        grad_scores = np.array(grad_scores)
        grad_probs = grad_scores / (grad_scores.sum() + 1e-6)

        gaussian_bias = np.random.normal(loc=0.5, scale=0.15, size=len(coords))
        gaussian_bias = np.clip(gaussian_bias, 0, 1)
        gaussian_probs = gaussian_bias / (np.sum(gaussian_bias) + 1e-6)

        final_probs = grad_probs + 0.1 * gaussian_probs
        final_probs = final_probs.astype(np.float64)
        final_probs = final_probs / final_probs.sum()

        chosen_indices = np.random.choice(len(coords), size=self.num_patches, replace=False, p=final_probs)

        compartments = []
        for idx in chosen_indices:
            x, y = coords[idx]
            patch = img_tensor[..., y:y+ps, x:x+ps]
            compartments.append(patch)

        return torch.stack(compartments)

class RGBLearner(nn.Module):
    def __init__(self):
        super(RGBLearner, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 2, kernel_size=3, padding=1),  # 注意这里输出2通道
        )

    def forward(self, x):
        out = self.encoder(x)
        out_combined = torch.cat([x, out], dim=1)
        return out_combined

class ChannelCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        proj_query = self.query_conv(feat1).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(feat2).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(feat2).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma * out + feat1
        return out

class WaveletBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.wavelet = WTConv2d(in_channels, out_channels, kernel_size=3, wt_levels=3, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.wavelet(x)))

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class WaveletDetailBranch(nn.Module):
    def __init__(self, in_channels, wave_dims=[256, 512, 1024, 2048]):
        super().__init__()
        self.stages = nn.ModuleList()

        for i in range(4):
            in_ch = in_channels if i == 0 else wave_dims[i - 1]
            out_ch = wave_dims[i]
            stage = nn.Sequential(
                WaveletBlock(in_ch, in_ch, downsample=(i != 0)),
                Conv2dBlock(in_ch, out_ch, downsample=False)
            )
            self.stages.append(stage)

        self.output_dims = wave_dims

    def forward(self, x_cat_patch): 
        feats = []
        x = x_cat_patch
        for stage in self.stages:
            x = stage(x)
            feats.append(x) 
        return feats

class DANetAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma_channel = nn.Parameter(torch.zeros(1))

        self.spatial_query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.spatial_key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.spatial_value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma_spatial = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(B, -1, H * W)
        proj_key = self.key_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        channel_attn = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        channel_out = torch.bmm(channel_attn, proj_value).view(B, C, H, W)
        channel_out = self.gamma_channel * channel_out + x

        proj_query = self.spatial_query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.spatial_key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        spatial_attn = torch.softmax(energy, dim=-1)
        proj_value = self.spatial_value(x).view(B, -1, H * W)
        spatial_out = torch.bmm(proj_value, spatial_attn.permute(0, 2, 1)).view(B, C, H, W)
        spatial_out = self.gamma_spatial * spatial_out + x
        return channel_out + spatial_out


class PatchTokenMixer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.attn = DANetAttention(in_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        attn_out = self.attn(x)
        return self.classifier(attn_out)

class AuxiliaryDecoder(nn.Module):
    def __init__(self, patch_channels=1, num_classes=45, num_patches=8):
        super().__init__()
        self.detail_branch = WaveletDetailBranch(in_channels=patch_channels * num_patches)
        self.token_mixer = PatchTokenMixer(2048, num_classes)

    def forward(self, patches): 
        B, N, C, H, W = patches.shape
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, N*C, H, W)

        feats = self.detail_branch(patches)
        logits = self.token_mixer(feats[-1])
        return logits, feats


class PaleoCellNet(nn.Module):
    def __init__(self, num_classes=45, use_pos_embedding=False, patch_size=128, num_patches=8):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding
        self.cbm_embedding_size = 1024

        self.rgb_learner = RGBLearner()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        patch_channels = 3 if use_pos_embedding else 1
        self.aux_decoder = AuxiliaryDecoder(patch_channels=patch_channels, num_classes=num_classes)

        self.attn_fuse = nn.ModuleList([
            ChannelCrossAttention(256),
            ChannelCrossAttention(512),
            ChannelCrossAttention(1024),
            ChannelCrossAttention(2048),
        ])

        self.final_fc1 = nn.Linear(2048, self.cbm_embedding_size)
        self.final_fc2 = nn.Linear(self.cbm_embedding_size, num_classes)

    def forward(self, x_img, x_patch):
        wavelet_logits, wavelet_feats = self.aux_decoder(x_patch)

        x_rgb = self.rgb_learner(x_img)  # 会resize为 [B, 3, 512, 512]
        x = self.backbone.conv1(x_rgb)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x1 = self.backbone.layer1(x)
        x2 = self.attn_fuse[1](self.backbone.layer2(x1), wavelet_feats[1])
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        
        x = self.backbone.avgpool(x4)
        x = x.view(x.size(0), -1)
        cls_cbm = self.final_fc1(x)
        cls_main = self.final_fc2(cls_cbm)

        return cls_main, cls_cbm, wavelet_logits
