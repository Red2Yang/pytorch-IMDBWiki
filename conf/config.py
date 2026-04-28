# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

IMG_DIR = BASE_DIR / "wiki"
MAT_FILE = IMG_DIR / "wiki.mat" #数据集内根目录有mat文件
PICKLE_FILE = BASE_DIR / "wiki_cleaned.pkl"
#IMG_DIR_STR = str(IMG_DIR)
PICKLE_FILE_STR = str(PICKLE_FILE)
#MAT_PATH = str(MAT_FILE)
IMG_DIR_STR = "E:/Dataset/wiki"
MAT_PATH = "E:/Dataset/wiki/wiki.mat"

# 年龄范围限制
MIN_AGE = 0
MAX_AGE = 100
MIN_FACE_SCORE = 0.0

# ========== 预训练权重路径 ==========
PRETRAINED_PATH = "resnet50-0676ba61.pth"

# ========== 默认训练参数 ==========
TASK = "both"  # 可选: "age", "gender", "both"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
VAL_SPLIT = 0.2 # 默认验证集在训练集占比
NUM_WORKERS = 4

# ========== 预处理参数 ==========
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ========== 数据增强参数 ==========
USE_RANDOM_HORIZONTAL_FLIP = True
USE_COLOR_JITTER = True
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1

# ========== 保存路径 ==========
SAVE_DIR = "./checkpoints"

# ========== 设备 ==========
DEVICE = None  # None自动选择可用设备