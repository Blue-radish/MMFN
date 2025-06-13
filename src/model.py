import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import io
import numpy as np
import pandas as pd
import subprocess
import torch
import random
import shutil
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
from qqdm import qqdm, format_str
import time
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoConfig, AutoTokenizer

class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_path, img_model_path):
        super(MultimodalClassifier, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.img_model = models.resnet50()
        self.img_model.load_state_dict(torch.load(img_model_path))
        self.img_model.fc = nn.Identity()
        img_model_dim = 2048

        # 从预训练的文本模型配置中获取隐藏层的大小
        config = AutoConfig.from_pretrained(text_model_path)
        text_model_dim = config.hidden_size

        # 计算Transformer的输入维度，它是文本特征和图像特征维度的总和
        transformer_dim = text_model_dim + img_model_dim

        self.attention_layer = nn.MultiheadAttention(embed_dim=img_model_dim, num_heads=8)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )

        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)  # Softmax函数，用于计算分类概率
        )

    def forward(self, input_ids, attention_mask, images):
        device = input_ids.device  # 确保设备一致

        # 处理文本特征
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        text_features = text_features[:, 0, :]  # 取出CLS标记的特征

        # 处理图像特征
        img_features_list = []
        for img_batch in images:
            img_features = []
            for img in img_batch:
                img_feature = self.img_model(img.unsqueeze(0).to(device))  # 处理单张图片
                img_features.append(img_feature)
            img_features = torch.stack(img_features, dim=0).to(device)  # 堆叠成张量
            # img_features, _ = self.attention_layer(img_features, img_features, img_features)  # 多头注意力
            img_features = img_features.mean(dim=0)  # 平均池化
            img_features_list.append(img_features)
        img_features = torch.cat(img_features_list, dim=0)  # 用cat

        # 合并文本和图像特征
        combined_features = torch.cat((text_features, img_features), dim=1).unsqueeze(0)

        # 通过Transformer编码器
        transformer_output = self.transformer_encoder(combined_features)

        # 分类
        logits = self.classifier(transformer_output.squeeze(0))

        return logits

class MultimodalDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像进行归一化
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = [i for i in self.data[idx]["text"]]
        images = self.data[idx]["img"]
        label = self.data[idx]["label"]

        if len(images) > 1:
            processed_images = []
            for img_data in images:
                img = Image.open(io.BytesIO(img_data))
                img = self.transform(img)
                processed_images.append(img)
            images = torch.stack(processed_images)
        elif len(images) == 1:
            img = Image.open(io.BytesIO(images[0]))
            img = self.transform(img)
            images = img.unsqueeze(0)
        else:
            img = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            images = img.unsqueeze(0)

        return text, images, label

