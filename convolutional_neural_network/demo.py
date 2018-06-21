# coding:utf-8

import numpy as np
import torch
import torchvision.transforms as transforms
from model import ConvNet
from PIL import Image


model_path = "./model.ckpt"
img_path = "./12.png"

model = ConvNet()
print "load pretrained model from %s" % model_path
model.load_state_dict(torch.load(model_path))

transformer = transforms.ToTensor()

image = Image.open(img_path).convert('L')
#image.resize((28, 28), Image.BILINEAR)
image = transformer(image)
image = image.view(1, *image.size())

model.eval()
output = model(image)

preds = torch.max(output, 1)[1]

print preds.item()
