import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model1 import Xue

# 数据增强与归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_test = transforms.ToTensor()

# 加载数据
full_train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True, transform=transform_test)

# 划分训练集和验证集（80/20）
train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 网络
xue = Xue()

# 损失函数 & 优化器 & 学习率调度器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(xue.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练配置
epoch = 30
writer = SummaryWriter("../logs_train1")
total_train_step = 0
total_val_step = 0

# 训练过程
for i in range(epoch):
    print(f"-------- 第{i+1}轮训练开始 --------")
    xue.train()
    for imgs, targets in train_loader:
        outputs = xue(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}，Loss：{loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 验证过程
    xue.eval()
    total_val_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            outputs = xue(imgs)
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()

            predictions = outputs.argmax(1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_samples

    print(f"验证集 Loss：{avg_val_loss:.4f}，准确率：{accuracy:.4f}")
    writer.add_scalar("val_loss", avg_val_loss, total_val_step)
    writer.add_scalar("val_accuracy", accuracy, total_val_step)
    total_val_step += 1

    # 每轮后更新学习率
    scheduler.step()

    # 保存模型
    torch.save(xue.state_dict(), f"xue_epoch_{i+1}.pth")
    print("模型已保存")

writer.close()
