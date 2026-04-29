import torch
import argparse
import cv2
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn

import conf.config as config

from utils.plt import draw_result

class AgeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        backbone = models.resnet50(weights=None)  # 不加载预训练
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone
    def forward(self, x):
        return self.backbone(x)

class GenderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        self.backbone = backbone
    def forward(self, x):
        return self.backbone(x)

class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)
    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        return age, gender

def load_model(model_path, task, device):
    if task == 'age':
        model = AgeEstimator()
    elif task == 'gender':
        model = GenderClassifier()
    else:
        model = AgeGenderNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, img_size=config.IMG_SIZE):
    """预处理输入图像，返回模型可接受的 tensor"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # 读取图像并转为 RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 与训练时验证集相同的预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])
    tensor = transform(image)
    return tensor.unsqueeze(0)   # 添加 batch 维度


def predict(model, image_tensor, task, device):
    """执行推理并返回结果"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        if task == 'age':
            age = model(image_tensor).item()
            return age, None
        elif task == 'gender':
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            gender = torch.argmax(probs, dim=1).item()
            return gender, probs[0].tolist()
        else:   # both
            age, gender_logits = model(image_tensor)
            age = age.item()
            probs = torch.softmax(gender_logits, dim=1)
            gender = torch.argmax(probs, dim=1).item()
            return age, gender, probs[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Predict age/gender from an image using trained model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth weight file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--task', type=str, default=config.TASK,
                        choices=['age', 'gender', 'both'], help='Prediction task')
    parser.add_argument('--img_size', type=int, default=config.IMG_SIZE,
                        help='Input image size (default from config)')
    args = parser.parse_args()

    # 设备
    if config.DEVICE is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 加载模型
    model = load_model(args.model_path, args.task, device)

    # 预处理图像
    image_tensor = preprocess_image(args.image_path, args.img_size)

    # 推理
    if args.task == 'age':
        age, _ = predict(model, image_tensor, args.task, device)
        print(f"Predicted age: {age:.1f} years")
        draw_result(args.image_path, age, None)
    elif args.task == 'gender':
        gender, probs = predict(model, image_tensor, args.task, device)
        gender_str = "Male" if gender == 1 else "Female"
        print(f"Predicted gender: {gender_str} (confidence: {max(probs):.3f})")
        gender_str = "Male" if gender == 1 else "Female"
        draw_result(args.image_path, None, gender_str, max(probs))
    else:   # both
        age, gender, probs = predict(model, image_tensor, args.task, device)
        gender_str = "Male" if gender == 1 else "Female"
        print(f"Predicted age: {age:.1f} years, gender: {gender_str} (confidence: {max(probs):.3f})")
        gender_str = "Male" if gender == 1 else "Female"
        draw_result(args.image_path, age, gender_str, max(probs))


if __name__ == '__main__':
    main()