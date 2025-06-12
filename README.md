# 一、项目概述  

本项目围绕 CIFAR-10 数据集展开，利用 Python 与深度学习框架，实现基础 CNN 模型原模型和改进模型训练与验证。    

# 二、环境搭建  

1、依赖工具    

Anaconda3：用于创建虚拟环境
  
2、虚拟环境配置

创建虚拟环境（Python 3.8）：

    conda create -n pytorch1.6 python=3.8

激活环境：  

    conda activate pytorch1.6  
    
安装依赖：  

    pip install torch==1.6.0 torchvision==0.7.0

# 三、数据集来源  

自动加载：利用 torchvision.datasets.CIFAR10 自动下载（需联网），代码如下：  

    import torchvision.datasets as datasets  
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True)  

# 四、模型训练与验证  

src/model.py：定义基础 CNN 模型（卷积层 + 全连接层），适配 CIFAR-10 输入尺寸。  

src/model1.py：改进CNN模型，层数变多  

src/train.py：CNN 模型训练代码，具体参数可以设置  

src/train1.py：改进CNN 模型训练代码，具体参数可以设置  

src/test.py：CNN 模型测试代码，测试图片文件可以切换    

src/test1.py：改进CNN 模型测试代码，测试图片文件可以切换    

ps：resnet代码没有在论文中使用！    



最后更新：2025 年 6 月 12 日    
维护者：Xue    
反馈渠道：联系 ahuangxue8@gmail.com






