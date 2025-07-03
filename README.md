# 第二十六届中国机器人及人工智能大赛
Source code for the "基于BERT预训练模型的虚假信息检测系统" project.

# Code

## Dataset

Weibo image and text dataset

## Data Extraction

data.ipynb

## Pre-trained model

- Download hfl_rbt6 and place it in ./models/hfl_rbt6
- Download resnet50 and place it in ./models/img_model

## Training

After processing the data using data.ipynb, run:

- train.ipynb (multi-modal, can handle multiple images)
- train_all.ipynb (multi-modal, if there are multiple images, only one will be used)
- train_text.ipynb (uni-modal, using only text information)

## WebUI

Run main.py

# Poster (基于BERT预训练模型的虚假信息检测系统)
![Poster.](./figure/WebUI.png)