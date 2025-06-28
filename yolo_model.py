import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import time
import os

MODEL_PTH = "cnn_model_0612_2_60.pth"

# ---------- 模型定義（和訓練時一樣） ----------
class EfficientCNN(nn.Module):
    def __init__(self, num_classes):
        super(EfficientCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (128, 128)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # (64, 64)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # (32, 32)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                     # (1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),  # 防止過擬合
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------- 載入模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['fall', 'normal']  # <-- 修改成你實際訓練的類別
num_classes = len(class_names)

#model = SimpleCNN(num_classes)
model = EfficientCNN(num_classes)
# 30 frame
model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
# 60 frame
#model.load_state_dict(torch.load("cnn_model_60frame_0527.pth", map_location=device))
model.to(device)
model.eval()

# ---------- 圖像轉換 ----------
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

def predict(pred_image):

    img_pil = Image.fromarray(pred_image)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        #_, pred = torch.max(output, 1)
        #label = class_names[pred.item()]
        probs = torch.softmax(output, dim=1)
        max_value, max_index = torch.max(probs, dim=1)
        top_value = max_value.item()
        
        pred = torch.argmax(probs, dim=1)
    
        label = class_names[pred.item()]
        #print(label, probs, max_value, top_value, pred)
        #print(label, probs, max_value, top_value, pred)
        print(label, top_value)
        return [label, top_value]

