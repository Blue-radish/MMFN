本仓库为论文 **《Fake Information Detection Method Based on BERT Pre-trained Model》** 的官方代码实现，对应第二十六届中国机器人及人工智能大赛参赛项目。  
The code implementation for the article *"Fake Information Detection Method Based on BERT Pre-trained Model"*.

论文链接（IEEE）：https://ieeexplore.ieee.org/abstract/document/11138502

---

## 📌 项目简介

MMFN是一个面向微博场景的多模态虚假信息检测系统，结合 **文本语义特征 + 图像视觉特征**，通过跨模态对齐机制实现对图文不一致、伪造内容的识别。

---

## 📁 仓库结构

```
MMFN
│  .gitignore
│  main.ipynb
│  main.py
│  README.md
│  requirements.txt
│
├─figure
│      bert.jpg
│      bert_1.png
│      WebUI.png
│
├─flagged
│      log.csv
│
├─models
│  ├─hfl_rbt6
│  │
│  └─img_model
│          resnet50.pth
│
├─notebook
│      data.ipynb
│      train.ipynb
│      train_all.ipynb
│      train_text.ipynb
│
├─src
│      model.py
│      train.py
│
└─temp                   # 临时文件（如爬取的微博图文）
```


---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```
建议使用 Python 3.10 与 GPU 环境（CUDA）。

### 数据预处理

data.ipynb

### 训练

- train.ipynb (multi-modal, can handle multiple images)
- train_all.ipynb (multi-modal, if there are multiple images, only one will be used)
- train_text.ipynb (uni-modal, using only text information)

### 启动WebUI

```bash
python main.py
```

## 🧠 模型结构

```
Text Encoder (hfl_rbt6)        Image Encoder (ResNet-50)
          ↓                               ↓
     Text Embedding                 Visual Embedding
          ↘                             ↙
        Multi-head Cross-modal Attention
                    ↓
               Fusion Vector
                    ↓
                Classifier
```

# Poster (基于BERT预训练模型的虚假信息检测系统)
![Poster.](./figure/WebUI.png)
