一、项目概述

本项目围绕 CIFAR-10 数据集展开，利用 Python 与深度学习框架，实现基础 CNN 模型原模型和改进模型训练与验证。

二、项目结构

rengongzhineng/  
├── .venv/              # Python 虚拟环境
├── .ipynb_checkpoints/ # Jupyter
├── dataset/            # 数据集目录  
│   └── cifar-10-python.tar.gz # CIFAR-10 原始数据集（通过脚本下载）  
├── models/             # 模型权重与配置（训练后生成，原模型和改进模型各30个）  
├── src/                # 核心代码目录  
│   ├── model.py        # 基础 CNN 模型定义  
│   ├── model1.py       # CNN 改进模型定义
|   ├── train.py        # 基础 CNN 模型训练
|   ├── train1.py       # CNN 改进模型训练
|   ├── test.py         # CNN 模型测试
|   ├── test1.py        # CNN 改进模型测试
│   ├── resnet18.py     # ResNet18 模型实现，没有采纳！！！
│   └── resnet18_train.py  # ResNet18 模型训练，没有采纳！！！
├── .gitignore          # Git 忽略规则配置  
├── requirements.txt    # 环境配置
└── README.md           # 项目说明文档（当前文件）  
