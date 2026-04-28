# Pytorch Training on IMDBWiki - A college course practice work.

一个校内训练项目，对`IMDB-WIKI（3GB）人脸数据集进行神经网络训练，实现对人的年龄和性别的识别。

配置要求：

1. N卡。
   - 如果是轻薄本的话，训练大概率会爆显存，只能用于结果展示
   - 1060以上显卡应该可以正常训练。
2. 有15GB以上空闲存储空间。

还有必备的Python虚拟环境！包括CUDA、Pytorch的安装，这里不再赘述。使用第三方库如下：

```txt
torch
torchvision
torchsummary
pillow
opencv-python
numpy
tqdm
scipy
```

## 数据集获取

[链接，估计下载奇慢无比](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

下载该3GB版本

![image](assets/dataset.png)

## 数据清洗



## 训练

```Python
python train.py
```