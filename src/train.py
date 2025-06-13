import time
import os
import io
from tqdm import tqdm
from qqdm import qqdm, format_str
import pandas as pd
import numpy as np
import random
import pickle
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import AdamW

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(7890)

# 从预训练模型加载分词器
tokenizer = AutoTokenizer.from_pretrained('../model/hfl_rbt6')  # 可以考虑换成'bert-base-chinese'，建议下载到本地使用

# 设置设备为GPU如果可用，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultimodalDataset(Dataset):
    def __init__(self, data_file):
        # 打开数据文件并加载数据
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        # 定义图像的预处理步骤
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图片大小以匹配模型输入
            transforms.ToTensor(),  # 将图片转换为PyTorch张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像进行归一化
        ])

    # 返回数据集的大小
    def __len__(self):
        return len(self.data)

    # 获取数据集中的一个样本
    def __getitem__(self, idx):
        # 获取文本、图像和标签
        text = [i for i in self.data[idx]["text"]]
        images = self.data[idx]["img"]
        label = self.data[idx]["label"]

        # 如果有图像，则打开第一张图像，否则创建一个全黑的图像
        if len(images) > 0:
            image = Image.open(io.BytesIO(images[0]))
        else:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 对图像进行预处理
        if self.transform:
            image = self.transform(image)

        # 返回文本、图像和标签
        return text, image, label

# 创建一个数据集实例
dataset = MultimodalDataset('train.pkl')

# 获取数据集中的第六个样本的tokens和labels
tokens, _, labels = dataset[5]
# 打印数据集的大小、第六个样本的tokens和labels
len(dataset), tokens, labels

# 定义collate_fn函数，用于在数据加载过程中对批次数据进行处理
def collate_fn(batch):
    batch_texts = []  # 创建一个列表用于存储批次中的文本数据
    batch_images = []  # 创建一个列表用于存储批次中的图像数据
    batch_labels = []  # 创建一个列表用于存储批次中的标签数据

    # 遍历批次中的每个样本
    for item in batch:
        batch_texts.append(item[0])  # 添加文本数据到batch_texts
        batch_images.append(item[1].unsqueeze(0))  # 添加图像数据到batch_images，并增加一个维度
        batch_labels.append(item[2])  # 添加标签数据到batch_labels

    # 使用tokenizer对文本数据进行预处理
    batch_texts = tokenizer.batch_encode_plus(batch_texts,
                                              truncation=True,
                                              padding=True,
                                              return_tensors='pt',
                                              is_split_into_words=True,
                                              max_length=512-2
                                             )
    # 将图像数据列表转换为一个批次的张量，并移除多余的维度
    batch_images = torch.stack(batch_images).squeeze(1)

    # 将标签数据转换为张量
    batch_labels = torch.tensor(batch_labels)

    # 将处理后的文本、图像和标签数据转移到指定的设备上（GPU或CPU）
    return batch_texts.to(device), batch_images.to(device), batch_labels.to(device)

loader = DataLoader(dataset=dataset,
                    batch_size=16,  # 指定批次大小
                    collate_fn=collate_fn,  # 指定collate_fn函数
                    shuffle=True,  # 数据打乱
                    drop_last=True)  # 如果最后一个批次不足batch_size，则丢弃
class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_path, img_model_path):
        super(MultimodalClassifier, self).__init__()
        # 加载预训练的文本模型
        self.text_model = AutoModel.from_pretrained(text_model_path)
        # 加载预训练的图像模型ResNet50
        self.img_model = models.resnet50(pretrained=img_model_path)
        # 将图像模型的最后一层（全连接层）替换为一个Identity层，这样就不会改变特征的维度
        self.img_model.fc = nn.Identity()
        # 设置图像模型的特征维度为2048，这是ResNet50的标准输出维度
        img_model_dim = 2048

        # 从预训练的文本模型配置中获取隐藏层的大小
        config = AutoConfig.from_pretrained(text_model_path)

        # 计算Transformer的输入维度，它是文本特征和图像特征维度的总和
        transformer_dim = config.hidden_size + img_model_dim
        
        # 定义一个Transformer编码器，用于处理融合后的文本和图像特征
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,  # 设置特征维度
                nhead=8,  # 设置多头注意力的头数
                dim_feedforward=2048,  # 前馈网络的维度
                dropout=0.1  # 设置dropout比率
            ),
            num_layers=6  # 设置Transformer编码器层的数量
        )
        
        # 定义一个分类头，它包含线性层和激活函数，用于最终的分类任务
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 512),  # 第一个线性层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, 2),  # 第二个线性层，输出维度为2，对应两个分类标签
            nn.Softmax(dim=1)  # Softmax函数，用于计算分类概率
        )

    # 定义前向传播过程
    def forward(self, input_ids, attention_mask, images):
        # 提取文本特征，特别是CLS标记的特征，它通常用于分类任务
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        text_features = text_features[:, 0, :]  # 取出CLS标记的特征

        # 提取图像特征
        img_features = self.img_model(images)

        # 将文本特征和图像特征进行融合
        combined_features = torch.cat((text_features, img_features), dim=1).unsqueeze(0)

        # 将融合后的特征通过Transformer编码器层
        transformer_output = self.transformer_encoder(combined_features)

        # 使用分类头对特征进行分类
        logits = self.classifier(transformer_output.squeeze(0))

        # 返回分类结果
        return logits
    
    num_epochs = 10
best_loss = 99
model_path = "..\model"
model = MultimodalClassifier('../model/hfl_rbt6', True).to(device)

criterion = nn.CrossEntropyLoss() # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
test_dataset = MultimodalDataset("test.pkl")
test_loader = DataLoader(dataset=test_dataset,
                batch_size=16,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True)  # 如果最后一个批次不足batch_size，则丢弃
train_metrics = []
test_metrics = []

best_test_loss = float('inf')  # 初始化loss为无穷大

for epoch in range(num_epochs):
    # 训练部分
    model.train()
    train_preds = []
    train_labels = []
    tot_loss = list()

    qqdm_loader = qqdm(loader, desc=format_str('bold', f'Training Epoch {epoch+1}'))
    for batch in qqdm_loader:
        batch_texts, batch_images, batch_labels = batch
        optimizer.zero_grad()
        outputs = model(batch_texts['input_ids'], batch_texts['attention_mask'], batch_images)
        loss = criterion(outputs, batch_labels)
        tot_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(batch_labels.cpu().numpy())

        # 更新qqdm进度条信息
        qqdm_loader.set_infos({
            'loss': f'{np.mean(tot_loss):.4f}',
        })

    mean_loss = np.mean(tot_loss)


    # 计算训练集指标
    train_metrics.append({
        'epoch': epoch + 1,
        'train_loss': mean_loss,
        'train_acc': accuracy_score(train_labels, train_preds),
        'train_precision': precision_score(train_labels, train_preds),
        'train_recall': recall_score(train_labels, train_preds),
        'train_f1': f1_score(train_labels, train_preds),
        'train_mAP': average_precision_score(train_labels, train_preds)
    })

    # 测试部分
    model.eval()
    test_preds = []
    test_labels = []
    test_loss = 0

    qqdm_test_loader = qqdm(test_loader, desc=format_str('bold', f'Testing Epoch {epoch+1}'))
    with torch.no_grad():
        for batch in qqdm_test_loader:
            batch_texts, batch_images, batch_labels = batch
            outputs = model(batch_texts['input_ids'], batch_texts['attention_mask'], batch_images)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())

            qqdm_test_loader.set_infos({
                'loss': f'{test_loss/(qqdm_test_loader.n+1):.4f}',
            })
    
    epoch_test_loss = test_loss / len(test_loader)

    # 计算测试集指标
    test_metrics.append({
        'epoch': epoch + 1,
        'test_loss': epoch_test_loss,
        'test_acc': accuracy_score(test_labels, test_preds),
        'test_precision': precision_score(test_labels, test_preds),
        'test_recall': recall_score(test_labels, test_preds),
        'test_f1': f1_score(test_labels, test_preds),
        'test_mAP': average_precision_score(test_labels, test_preds)
    })

    # 根据测试集loss保存最佳模型
    if epoch_test_loss < best_test_loss:
        best_test_loss = epoch_test_loss
        torch.save(model, os.path.join(model_path, 'FN_best.pth'))

# 保存指标到CSV
pd.DataFrame(train_metrics).to_csv('train_metrics.csv', index=False)
pd.DataFrame(test_metrics).to_csv('test_metrics.csv', index=False)