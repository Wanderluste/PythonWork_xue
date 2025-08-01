import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from model import *


#数据准备
#训练数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
#测试数据集
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
#长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

#用dataloader来加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#创建网络模型有cuda类型
class Xue(nn.Module):
    def __init__(self):
        super(Xue, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x


xue = Xue()
xue = xue.cuda()

#损失函数有cuda
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(xue.parameters(), lr=learning_rate)

#设置训练的网络参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练轮数
epoch = 30
#添加画图
writer = SummaryWriter("../logs_train")
i = 0
for i in range(epoch):
    print("--------第{}轮训练开始---------".format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = xue(imgs)
        loss = loss_fn(outputs, targets)
#优化器调用
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss, total_train_step)

    #测试步骤
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = xue(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率accuracy:{}".format(total_accuracy.float()/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy.float()/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(xue, 'xue_{}.pth'.format(i))
    print("模型已保存")

writer.close()


