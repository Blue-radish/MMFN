import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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
import gradio as gr
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoConfig, AutoTokenizer
import requests
import re
from bs4 import BeautifulSoup

from src.model import MultimodalClassifier, MultimodalDataset

tokenizer = AutoTokenizer.from_pretrained('./model/hfl_rbt6')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_web(url):
    headers={
        "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
        "Cookie":
    "XSRF-TOKEN=deleted; expires=Thu, 01-Jan-1970 00:00:01 GMT; Max-Age=0; path=/; domain=.weibo.cn; HttpOnly",
    "Referer":
    url
    }
    
    resp=requests.get(url,headers=headers)
    
    obj=re.compile(r'"text":(?P<text>.*?)"textLength"',re.S)
    obj1=re.compile(r"[\u4e00-\u9fa5,。“”]",re.S)
    obj2=re.compile(r'"pics": .*?"url": "(?P<pic_url>.*?)",',re.S)
    results=obj.finditer(resp.text)
    chinese_text = ""
    for result in results:
        text=result.group("text")
        matches = obj1.finditer(text)
        for match in matches:
            chinese_text += match.group()
    pics=obj2.finditer(resp.text)
    pic_url=""
    for pic in pics:
        pic_url=pic.group("pic_url")
    print(pic_url)
    pic_name=pic_url.split("/")[-1]
    # txt_name=pic_url.split("/")[-1].split(".")[0]+".txt"
    txt_name="text.txt"
    
    pic_resp=requests.get(pic_url,headers=headers)
    
    file_path_pic = rf".\temp\imgs"
    file_path_txt = rf".\temp"
    if os.path.exists(file_path_txt):
        shutil.rmtree(file_path_txt)
        os.makedirs(file_path_pic)

    with open(os.path.join(file_path_pic, pic_name),"wb") as f:
        f.write(pic_resp.content)
    print("pic_over")
    with open(os.path.join(file_path_txt, txt_name),'w', encoding='utf-8') as f:
        f.write(chinese_text)
    print("txt_over")
    resp.close()
    pic_resp.close()

def web_data(url):
    get_web(url)
    img_path=r".\temp\imgs"
    for file in os.listdir(img_path):
        name = file.split('.')[0]
        file_path = os.path.join(img_path, file)
        output_path = os.path.join(img_path, f"{name}.jpg")
        with Image.open(file_path) as i:
            img = i.convert('RGB')
        os.remove(file_path)
        img.save(output_path, 'JPEG')
            
    with open(r".\temp\text.txt", 'r', encoding='utf-8') as file:
        text = file.read()
    arr=[{'text': text, 'img': [], 'label': None}]
    for img in os.listdir(img_path):
        with open(os.path.join(img_path, img), 'rb') as file:
            img_data = file.read()
            arr[0]['img'].append(img_data)
    with open(r'temp\test.pkl', 'wb') as file:
        pickle.dump(arr, file)
        
    img = Image.open(output_path)
    with open(r".\temp\text.txt", 'r', encoding='utf-8') as file:
        text = file.read()
    return text, img

def data(imgs_path, text, model_name):
    file_path_pic = rf".\temp\imgs"
    file_path_txt = rf".\temp"
    img_path=r".\temp\imgs"
    if os.path.exists(file_path_txt):
        shutil.rmtree(file_path_txt)
        os.makedirs(file_path_pic)
    for file_path in imgs_path:
        name = file_path.split('\\')[-1].split('.')[0]
        output_path = os.path.join(r".\temp\imgs", f"{name}.jpg")
        with Image.open(file_path) as i:
            img = i.convert('RGB')
        img.save(output_path, 'JPEG')
    txt_name="text.txt"
    with open(os.path.join(file_path_txt, txt_name),'w', encoding='utf-8') as f:
        f.write(text)

    with open(r".\temp\text.txt", 'r', encoding='utf-8') as file:
        text = file.read()
    arr=[{'text': text, 'img': [], 'label': None}]
    for img in os.listdir(img_path):
        with open(os.path.join(img_path, img), 'rb') as file:
            img_data = file.read()
            arr[0]['img'].append(img_data)
    with open(r'temp\test.pkl', 'wb') as file:
        pickle.dump(arr, file)
    
    return FN(model_name)

def collate_fn(batch):
    batch_texts = []
    batch_images = []
    for item in batch:
        batch_texts.append(item[0])
        batch_images.append(item[1]) # 将每个样本的所有图片作为一个列表添加
    # 对文字进行预处理
    batch_texts = tokenizer.batch_encode_plus(batch_texts,
                                         truncation=True,
                                         padding=True,
                                         return_tensors='pt',
                                         is_split_into_words=True,
                                         max_length=512-2
                                        )

    # 不在这里转换图像数据为张量，在模型的forward方法中处理
    return batch_texts.to(device), batch_images

# def collate_fn(batch):
#     batch_texts = []
#     batch_images = []
#     # batch_labels = []

#     # 遍历批次中的每个样本
#     for item in batch:
#         batch_texts.append(item[0])
#         batch_images.append(item[1].unsqueeze(0))
#         # batch_labels.append(item[2])

#     batch_texts = tokenizer.batch_encode_plus(batch_texts,
#                                               truncation=True,
#                                               padding=True,
#                                               return_tensors='pt',
#                                               is_split_into_words=True,
#                                               max_length=512-2
#                                              )
#     batch_images = torch.stack(batch_images).squeeze(1)

#     # batch_labels = torch.tensor(batch_labels)

#     # return batch_texts.to(device), batch_images.to(device), batch_labels.to(device)
#     return batch_texts.to(device), batch_images.to(device)

def FN(model_name):
    print(model_name)
    test_path = './temp/test.pkl'
    model_path = f'./model/{model_name}'
    with open(test_path, 'rb') as file:
        test_data = pickle.load(file)
    model = torch.load(model_path, map_location=device)
    test_dataset = MultimodalDataset(test_path)
    test_loader = DataLoader(dataset=test_dataset,
                    batch_size=1,
                    collate_fn=collate_fn,
                    shuffle=True,
                    drop_last=True)  # 如果最后一个批次不足batch_size，则丢弃
    # 设置模型为评估模式
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        for batch in test_loader:
            batch_texts, batch_images = batch
            outputs = model(batch_texts['input_ids'], batch_texts['attention_mask'], batch_images)
            _, predicted = torch.max(outputs, 1)
            # print(f"假新闻: {outputs[0][0].item()*100:.2f}%\t真新闻: {outputs[0][1].item()*100:.2f}%")
            return f"假新闻概率: {outputs[0][0].item()*100:.4f}%\t\t真新闻概率: {outputs[0][1].item()*100:.4f}%"

def training(name='FN', num_epoch=3, bert="hfl_rbt6", lr=1e-5, batch_size=8, train_data='', test_data=''):
    def collate_fn(batch):
        batch_texts = []
        batch_images = []
        batch_labels = []
        for item in batch:
            batch_texts.append(item[0])
            batch_images.append(item[1]) # 将每个样本的所有图片作为一个列表添加
            batch_labels.append(item[2])
        # 对文字进行预处理
        batch_texts = tokenizer.batch_encode_plus(batch_texts,
                                             truncation=True,
                                             padding=True,
                                             return_tensors='pt',
                                             is_split_into_words=True,
                                             max_length=512-2
                                            )
        # 标签
        batch_labels = torch.tensor(batch_labels)
        # 不在这里转换图像数据为张量，在模型的forward方法中处理
        return batch_texts.to(device), batch_images, batch_labels.to(device)
    
    lr = float(lr)
    dataset = MultimodalDataset(train_data)
    loader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    shuffle=True,
                    drop_last=True)
    best_loss = 99
    model_path = "./model" # 模型存放路径
    model = MultimodalClassifier(f'./model/{bert}', './model/img_model/resnet50.pth').to(device)
    criterion = nn.CrossEntropyLoss() # 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    progress=gr.Progress()
    for epoch in progress.tqdm(range(1, num_epoch+1), desc='训练中...'):
        tot_loss = []
        for batch in loader:
            batch_texts, batch_images, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(batch_texts['input_ids'], batch_texts['attention_mask'], batch_images)
            loss = criterion(outputs, batch_labels)
            tot_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(tot_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, os.path.join(model_path, f'{name}_best.pth'))
        torch.save(model, os.path.join(model_path, f'{name}_last.pth'))
        
    if test_data != '':
        model = torch.load(os.path.join(model_path, f'{name}_best.pth'), map_location=device)
        test_dataset = MultimodalDataset(test_data)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=16,
                                 collate_fn=collate_fn,
                                 shuffle=True,
                                 drop_last=True)  # 如果最后一个批次不足batch_size，则丢弃
        y_true = []
        y_scores = []
        y_pred = []
    
        model.eval()
        with torch.no_grad():
            progress = gr.Progress()
            for batch in progress.tqdm(test_loader, desc='测试中...'):
                batch_texts, batch_images, batch_labels = batch
                outputs = model(batch_texts['input_ids'], batch_texts['attention_mask'], batch_images)
                y_true.extend(batch_labels.tolist())
                y_scores.extend(outputs[:, 1].tolist())  # 正类标签为1
                y_pred.extend((outputs[:, 1] > 0.5).int().tolist())  # 阈值为0.5
    
        # 计算正确率
        correct_predictions = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
        accuracy = correct_predictions / len(y_true)
    
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
        plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('误报率')
        plt.ylabel('真正例率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        return f"best_loss={best_loss} \t\t last_loss={mean_loss}", plt.gcf(), accuracy
        # print(accuracy)
        # return f"best_loss={best_loss} \t\t last_loss={mean_loss}", plt.gcf(), 86.8
    else:
        return f"best_loss={best_loss} \t\t last_loss={mean_loss}", None, None

with gr.Blocks() as web:
    with gr.Tab(label="inference"):
        with gr.Row(): # 水平
            with gr.Column(scale=3): # 垂直
                model_list = [f for f in os.listdir("./model") if f.endswith('.pth')]
                inf_model = gr.Dropdown(choices=model_list, label="选择模型")
            with gr.Column(scale=1):
                inf_run = gr.Button(variant="primary")
        with gr.Row(): # 水平
            with gr.Column(scale=2):
                inf_imgs = gr.File(file_count="multiple", type="filepath", label="Upload Images")
            with gr.Column(scale=3):
                inf_text = gr.Textbox(label="文字")
        with gr.Row(): # 水平
            with gr.Column(scale=1):
                gr.Image(value="./model/img_model/bert.jpg")
            with gr.Column(scale=3):
                inf_output = gr.Textbox(label="output")
        inf_run.click(fn=data, inputs=[inf_imgs, inf_text, inf_model], outputs=inf_output)
    
    with gr.Tab(label="web_inference"):
        with gr.Row(): # 水平
            with gr.Column(scale=5): # 垂直
                web_http = gr.Textbox(label="input")
            with gr.Column(scale=3): # 垂直
                model_list = [f for f in os.listdir("./model") if f.endswith('.pth')]
                web_model = gr.Dropdown(choices=model_list, label="选择模型")
            with gr.Column(scale=1):
                web_get = gr.Button(variant="primary", value="Get")
                web_run = gr.Button(variant="primary")
        web_output = gr.Textbox(label="output")
        with gr.Row():
            with gr.Column(scale=2):
                web_img = gr.Image()
            with gr.Column(scale=3):
                web_text = gr.Textbox(label="文字")
        web_run.click(fn=FN, inputs=web_model, outputs=web_output)
        web_get.click(fn=web_data, inputs=web_http, outputs=[web_text, web_img])
        
    with gr.Tab(label="train"):
        with gr.Row(): # 水平
            with gr.Column(scale=2):
                tr_name = gr.Textbox(label="实验名称")
            with gr.Column(scale=2):
                tr_epoch = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=3,
                    step=1,
                    label="epoch",
                    interactive=True,
                )
            with gr.Row(): # 水平
                with gr.Column(scale=3):
                    tr_run = gr.Button(variant="primary", value="训练")
                with gr.Column(scale=3):
                    tr_bert = gr.Radio(choices=["hfl_rbt6", "bert-chinese"], label="bert底模", value="hfl_rbt6")
        with gr.Row(): # 水平
            with gr.Column(scale=3):
                tr_lr = gr.Textbox(label="学习率", value=0.00001)
            with gr.Column(scale=3):
                tr_batch_size = gr.Slider(
                    minimum=1,
                    maximum=256,
                    value=8,
                    step=1,
                    label="batch_size",
                    interactive=True,
                )
        with gr.Row(): # 水平
            with gr.Column():
                train_data = gr.Textbox(label="训练集路径")
            with gr.Column():
                test_data = gr.Textbox(label="测试集路径")
        with gr.Row(): # 水平
            tr_info = gr.Textbox(label="输出信息")
        with gr.Row(): # 水平
            with gr.Column():
                plot_output = gr.Plot(label="ROC Curve")
            with gr.Column():
                test_output = gr.Textbox(label="正确率")
        tr_run.click(fn=training, inputs=[tr_name, tr_epoch, tr_bert, tr_lr, tr_batch_size, train_data, test_data], 
                     outputs=[tr_info, plot_output, test_output])

web.launch()