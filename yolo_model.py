import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import time
import os

import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

MODEL_PTH = "cnn_model_0612_2_60.pth"
#MODEL_PTH = "pose_model.pth"

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

class_names = ['fall', 'normal']  # <-- 修改成你實際訓練的類別

def load_model():
    # ---------- 載入模型 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    print(f"✅ Using device: {device}")
    num_classes = len(class_names)
    #model = SimpleCNN(num_classes)
    model = EfficientCNN(num_classes)
    # 30 frame
    print("load_model", MODEL_PTH )
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
    return model,transform,device

def predict(model,transform,device,pred_image):

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

def train(model_file, data_dir):
    img_size = 640
    batch_size = 32
    num_epochs = 30
    # ---------------- GPU 設定 ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    # -----------------------------------------

    # 資料預處理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # 自動轉 [0,255] -> [0.0,1.0]
    ])

    # 載入資料集
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    print("Label classes:", full_dataset.classes)

    # 切分 train/val
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)    

    model = EfficientCNN(num_classes).to(device)

    # Loss 與 Optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 訓練
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.4f}")

    # 驗證
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print("Validation Accuracy:", val_acc)

    output_model_name = model_file
    # 儲存模型
    torch.save(model.state_dict(),output_model_name)
    print("✅ 模型已儲存為",output_model_name)