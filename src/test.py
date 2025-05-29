import torch
import torchvision
from PIL import Image
import model

image_path = "../image/img_11.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

Model = torch.load("../models/xue_25.pth")
print(Model)
image = torch.reshape(image, (1, 3, 32, 32))
Model.eval()
with torch.no_grad():
    output = Model(image)
print(output)
print(output.argmax(1))

