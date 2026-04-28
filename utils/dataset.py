import pickle
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

import conf.config as config

class IMDBWikiDataset(Dataset):
    """IMDB-WIKI 数据集，自动适配行式/列式，自动检测图像字段，过滤无效样本"""
    def __init__(self, pkl_path, transform=None, task='age'):
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

        # 格式识别
        if isinstance(raw, dict):
            first_val = next(iter(raw.values()))
            if isinstance(first_val, list):
                self._mode = 'column'
                self._data = raw
                self._len = len(self._data[list(self._data.keys())[0]])
                print(f"Loaded column-wise dict: {self._len} samples, keys: {list(self._data.keys())}")
            elif isinstance(first_val, dict):
                self._mode = 'row_list'
                self._data = list(raw.values())
                self._len = len(self._data)
                print(f"Converted row-wise dict to list: {self._len} samples")
            else:
                raise TypeError(f"Unsupported dict value type: {type(first_val)}")
        elif isinstance(raw, list):
            self._mode = 'row_list'
            self._data = raw
            self._len = len(self._data)
            print(f"Loaded row-wise list: {self._len} samples")
        else:
            raise TypeError(f"Unexpected data type: {type(raw)}")

        # 自动检测图像字段名
        possible_image_keys = ['image', 'img', 'photo', 'samples', 'data', 'face', 'image_data', 'picture']
        self.image_key = None

        if self._mode == 'column':
            for key in possible_image_keys:
                if key in self._data:
                    self.image_key = key
                    break
            if self.image_key is None:
                for key in self._data.keys():
                    if key not in ['age', 'gender', 'dob', 'birth']:
                        self.image_key = key
                        break
        else:
            if self._len > 0:
                sample0 = self._data[0]
                for key in possible_image_keys:
                    if key in sample0:
                        self.image_key = key
                        break
                if self.image_key is None:
                    for key in sample0.keys():
                        if key not in ['age', 'gender', 'dob', 'birth']:
                            self.image_key = key
                            break

        if self.image_key is None:
            raise KeyError("Cannot identify image field. Available keys: " +
                           (str(list(self._data.keys())) if self._mode == 'column' else str(list(sample0.keys()))))

        print(f"Detected image field: '{self.image_key}'")

        # 构建有效样本索引（过滤图像为 None 或加载失败的样本）
        self.valid_indices = []
        for idx in range(self._len):
            if self._mode == 'column':
                img_val = self._data[self.image_key][idx]
            else:
                img_val = self._data[idx].get(self.image_key)

            if img_val is not None:
                if isinstance(img_val, str) and not os.path.exists(img_val):
                    continue
                self.valid_indices.append(idx)

        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid samples found (all image fields are None or missing).")

        print(f"Filtered: {len(self.valid_indices)} valid samples out of {self._len} (removed {self._len - len(self.valid_indices)})")
        self._len = len(self.valid_indices)

        self.transform = transform
        self.task = task

    def __len__(self):
        return self._len

    def _load_image(self, img_val):
        """加载图像：支持 numpy 数组或文件路径"""
        if isinstance(img_val, np.ndarray):
            image = img_val
        elif isinstance(img_val, str):
            image = cv2.imread(img_val)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {img_val}")
        else:
            raise TypeError(f"Unsupported image type: {type(img_val)}")

        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        if self._mode == 'column':
            sample = {key: self._data[key][real_idx] for key in self._data.keys()}
        else:
            sample = self._data[real_idx]

        img_val = sample.get(self.image_key)
        if img_val is None:
            raise RuntimeError(f"Sample {real_idx} has None image after filtering?")

        image = self._load_image(img_val)
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        if self.task == 'age':
            age = sample.get('age')
            if age is None:
                raise KeyError(f"Sample {real_idx} missing 'age'")
            return image, torch.tensor(float(age), dtype=torch.float32)
        elif self.task == 'gender':
            gender_val = sample.get('gender')
            if gender_val is None:
                raise KeyError(f"Sample {real_idx} missing 'gender'")
            if isinstance(gender_val, str):
                gender = 0 if gender_val.upper() == 'F' else 1
            else:
                gender = int(gender_val)
            return image, torch.tensor(gender, dtype=torch.long)
        else:  # both
            age = sample.get('age')
            gender_val = sample.get('gender')
            if age is None or gender_val is None:
                raise KeyError(f"Sample {real_idx} missing 'age' or 'gender'")
            age_t = torch.tensor(float(age), dtype=torch.float32)
            if isinstance(gender_val, str):
                gender = 0 if gender_val.upper() == 'F' else 1
            else:
                gender = int(gender_val)
            gender_t = torch.tensor(gender, dtype=torch.long)
            return image, (age_t, gender_t)


def get_train_transform():
    """训练数据增强 pipeline"""
    transform_list = [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    ]
    if config.USE_RANDOM_HORIZONTAL_FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())
    if config.USE_COLOR_JITTER:
        transform_list.append(transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
            saturation=config.COLOR_JITTER_SATURATION,
            hue=config.COLOR_JITTER_HUE
        ))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])
    return transforms.Compose(transform_list)


def get_val_transform():
    """验证集预处理 pipeline"""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])