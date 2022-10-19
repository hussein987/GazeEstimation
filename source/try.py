import cv2
from PIL import Image
import numpy as np
import dlib
import torch
from CNN import CNN1
import torchvision.transforms as T

transforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])


def draw_point(path, points):
    img = cv2.imread(path)
    x, y = points[0], points[1]
    print(x, y)
    # cv2.circle(img, (x*80, y*80), 2, (0, 255, 0), 2)
    # cv2.circle(img, (67, 83), 2, (255, 0, 0), 2)
    c_x, c_y = 45, 40
    cv2.arrowedLine(img, (c_x, c_y), (c_x+x*80, c_y+y*80), (0, 255, 0), 2)

    cv2.imshow('name', img)
    cv2.waitKey(0)


model = CNN1()
model.load_state_dict(torch.load('../models/model2.pt', map_location=torch.device('cpu')))
img_path = 'real.jpg'
img = Image.open(img_path).convert("RGB")
img = transforms(img)
out = model(img.unsqueeze(0))
# draw(img_path, [out['label1'], out['label2']])
draw_point(img_path, [out['label1'], out['label2']])