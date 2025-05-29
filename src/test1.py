import torch
import torchvision
from PIL import Image
from model1 import Xue

image_path = "../image/img_11.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

model = Xue()
model.load_state_dict(torch.load("../models/xue_epoch_20.pth"))  # 替换为你的模型路径
model.eval()  # 切换到推理模式（关闭Dropout/BatchNorm等）
print(model)
image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))

