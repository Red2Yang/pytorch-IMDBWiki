import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import conf.config as config
from utils.dataset import *
from utils.models import *

from conf.config import PRETRAINED_PATH

def train_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Training'):
        if task == 'age':
            images, ages = batch
            images, ages = images.to(device), ages.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
        elif task == 'gender':
            images, genders = batch
            images, genders = images.to(device), genders.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, genders)
        else:
            images, (ages, genders) = batch
            images = images.to(device)
            ages = ages.to(device).unsqueeze(1)
            genders = genders.to(device)
            optimizer.zero_grad()
            pred_ages, pred_genders = model(images)
            loss_age = criterion[0](pred_ages, ages)
            loss_gender = criterion[1](pred_genders, genders)
            loss = loss_age + loss_gender
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            if task == 'age':
                images, ages = batch
                images, ages = images.to(device), ages.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, ages)
            elif task == 'gender':
                images, genders = batch
                images, genders = images.to(device), genders.to(device)
                outputs = model(images)
                loss = criterion(outputs, genders)
            else:
                images, (ages, genders) = batch
                images = images.to(device)
                ages = ages.to(device).unsqueeze(1)
                genders = genders.to(device)
                pred_ages, pred_genders = model(images)
                loss_age = criterion[0](pred_ages, ages)
                loss_gender = criterion[1](pred_genders, genders)
                loss = loss_age + loss_gender
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

# ------------------------- 主函数 -------------------------
def main():
    parser = argparse.ArgumentParser(description='Train IMDB-WIKI on PyTorch 2')
    parser.add_argument('--pkl_path', type=str, required=True, help='Path to .pkl file')
    parser.add_argument('--task', type=str, default=config.TASK, choices=['age', 'gender', 'both'])
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--val_split', type=float, default=config.VAL_SPLIT)
    parser.add_argument('--img_size', type=int, default=config.IMG_SIZE)
    parser.add_argument('--save_dir', type=str, default=config.SAVE_DIR)
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 设备
    if config.DEVICE is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.DEVICE)
    print(f'Using device: {device}')

    # 数据增强
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # 加载完整数据集（使用训练transform，仅用于获取样本总数和索引）
    full_dataset = IMDBWikiDataset(args.pkl_path, transform=train_transform, task=args.task)
    total_samples = len(full_dataset)
    val_size = int(total_samples * args.val_split)
    train_size = total_samples - val_size

    # 划分索引
    train_indices, val_indices = random_split(range(total_samples), [train_size, val_size])
    # 重新构建数据集实例，分别应用不同的transform
    train_dataset = IMDBWikiDataset(args.pkl_path, transform=train_transform, task=args.task)
    val_dataset = IMDBWikiDataset(args.pkl_path, transform=val_transform, task=args.task)
    # 使用Subset按索引采样
    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f'Training samples: {train_size}, Validation samples: {val_size}')

    # 模型及损失函数
    if args.task == 'age':
        model = AgeEstimator(PRETRAINED_PATH=config.PRETRAINED_PATH)
        criterion = nn.MSELoss()
    elif args.task == 'gender':
        model = GenderClassifier(PRETRAINED_PATH=config.PRETRAINED_PATH)
        criterion = nn.CrossEntropyLoss()
    else:
        model = AgeGenderNet(PRETRAINED_PATH=config.PRETRAINED_PATH)
        criterion = (nn.MSELoss(), nn.CrossEntropyLoss())

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.task)
        val_loss = validate_epoch(model, val_loader, criterion, device, args.task)
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f'best_model_{args.task}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved to {model_path}')

    print('Training completed.')

if __name__ == '__main__':
    main()