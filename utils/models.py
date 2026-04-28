import os
import torch
import torch.nn as nn
from torchvision import models

import conf.config as config

class AgeEstimator(nn.Module):
    """年龄估计模型（ResNet50回归）"""
    def __init__(self, PRETRAINED_PATH):
        super().__init__()
        # 创建不加载预训练权重的模型
        backbone = models.resnet50(weights=None)
        
        if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
            state_dict = torch.load(PRETRAINED_PATH, map_location='cpu')
            # 由于后面要替换fc层，使用strict=False避免fc层键名不匹配
            backbone.load_state_dict(state_dict, strict=False)
            print(f"AgeEstimator: Loaded pretrained weights from {PRETRAINED_PATH}")
        else:
            print("AgeEstimator: No pretrained weights loaded, auto downloading...")
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone
    
    def forward(self, x):
        return self.backbone(x)


class GenderClassifier(nn.Module):
    """性别分类模型（ResNet50二分类）"""
    def __init__(self, PRETRAINED_PATH):
        super().__init__()
        backbone = models.resnet50(weights=None)
        
        if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
            state_dict = torch.load(PRETRAINED_PATH, map_location='cpu')
            backbone.load_state_dict(state_dict, strict=False)
            print(f"GenderClassifier: Loaded pretrained weights from {PRETRAINED_PATH}")
        else:
            print("GenderClassifier: No pretrained weights loaded, auto downloading...")
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        self.backbone = backbone
    
    def forward(self, x):
        return self.backbone(x)


class AgeGenderNet(nn.Module):
    """联合模型：共享特征提取器，双头输出年龄和性别"""
    def __init__(self, PRETRAINED_PATH):
        super().__init__()
        backbone = models.resnet50(weights=None)
        
        if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
            state_dict = torch.load(PRETRAINED_PATH, map_location='cpu')
            backbone.load_state_dict(state_dict, strict=False)
            print(f"AgeGenderNet: Loaded pretrained weights from {PRETRAINED_PATH}")
        else:
            print("AgeGenderNet: No pretrained weights loaded, auto downloading...")
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()      # 移除原始分类头
        self.backbone = backbone
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)
    
    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        return age, gender