from ultralytics import YOLO
import cv2
import math
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import datetime
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------- U-Net模型定義 ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        b = self.bottleneck(self.pool4(d4))

        u4 = self.up4(b)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.dec4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1)

        return self.final_conv(u1)

# ---------------------------- 輔助函數 ----------------------------
def calculate_center(x1, y1, x2, y2):
    """計算中心點"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def calculate_distance(point1, point2):
    """計算兩點距離"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def save_quadrant_images(quadrants, image_path):
    """保存象限圖片到指定資料夾"""
    # 設定保存路徑
    base_save_path = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Quadrant_division"
    
    # 獲取檔案名稱（不含副檔名）
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 創建以檔案名稱命名的資料夾
    folder_path = os.path.join(base_save_path, filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"創建資料夾：{folder_path}")
    
    # 獲取當前日期
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 保存每個象限圖片
    saved_files = []
    for q_name, q_image in quadrants.items():
        save_filename = f"{q_name}_{current_date}.jpg"
        save_filepath = os.path.join(folder_path, save_filename)
        success = cv2.imwrite(save_filepath, q_image)
        if success:
            saved_files.append(save_filepath)
            print(f"已保存：{save_filepath}")
    
    return saved_files

def process_image_with_unet(image, model):
    """使用U-Net處理圖片"""
    # 調整圖片大小為512x512
    image_resized = cv2.resize(image, (512, 512))
    
    # 轉換為PyTorch張量
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    # 進行預測
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1)
        pred = pred.squeeze().numpy()
    
    return pred, image_resized

def draw_zone1(image, od_center, fovea_center=None, od_radius=None):
    image_with_zone = image.copy()
    
    # 繪製視盤中心點（綠色）
    cv2.circle(image_with_zone, 
              (int(od_center[0]), int(od_center[1])), 
              5, (0, 255, 0), -1)
    cv2.putText(image_with_zone, 'Optic Disc (1)', 
                (int(od_center[0] - 60), int(od_center[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if fovea_center is not None:
        # 計算視盤中心點和黃斑點之間的歐氏距離
        distance = calculate_distance(od_center, fovea_center)
        
        # ZONE1 的半徑是兩點距離的2倍
        zone1_radius = distance * 2
        
        # 繪製 ZONE1 範圍（綠色圓）
        cv2.circle(image_with_zone, 
                  (int(od_center[0]), int(od_center[1])), 
                  int(zone1_radius), (0, 255, 0), 2)
        
        # 繪製黃斑點（紅色）
        cv2.circle(image_with_zone, 
                  (int(fovea_center[0]), int(fovea_center[1])), 
                  5, (0, 0, 255), -1)
        cv2.putText(image_with_zone, 'Fovea (0)', 
                    (int(fovea_center[0] - 50), int(fovea_center[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 繪製視盤中心到黃斑點的連接線（紅色）
        cv2.line(image_with_zone, 
                (int(od_center[0]), int(od_center[1])),
                (int(fovea_center[0]), int(fovea_center[1])), 
                (0, 0, 255), 2)

    return image_with_zone

def calculate_distance(point1, point2):
    """計算兩點間的歐氏距離"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main():
    # 載入模型
    yolo_model = YOLO(r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\ROP-zones\runs\segment\ROP6\weights\best.pt")
    unet_model = UNet()
    checkpoint = torch.load(r"D:\ROP\best_vessel_cnn.pth")
    unet_model.load_state_dict(checkpoint['model_state_dict'])  # 只載入模型權重
    unet_model.eval()

    # 創建檔案選擇對話框
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="選擇眼底影像")
    
    if not image_path:
        print("未選擇檔案")
        return

    # 讀取原始圖片
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("無法讀取圖片")
        return

    # 執行YOLOv8預測
    results = yolo_model(original_image)
    
    # 處理YOLO結果，獲取視盤和黃斑點位置
    od_center = None
    fovea_center = None
    od_radius = None
    
    for r in results:
        for i, cls in enumerate(r.boxes.cls):
            box = r.boxes.xyxy[i].cpu().numpy()  # 確保轉換為 numpy array
            if int(cls) == 1:  # 視盤類別
                od_center = calculate_center(box[0], box[1], box[2], box[3])
                od_radius = (box[2] - box[0]) / 2  # 計算視盤半徑
            elif int(cls) == 0:  # 黃斑點類別
                fovea_center = calculate_center(box[0], box[1], box[2], box[3])
    
    if od_center is None:
        print("未檢測到視盤")
        return

    # 在原始圖片上繪製 ZONE1
    original_with_zone1 = draw_zone1(original_image, od_center, fovea_center, od_radius)

    # 使用U-Net進行血管分割
    vessel_mask, resized_image = process_image_with_unet(original_image, unet_model)
    
    # 將遮罩轉換為彩色圖像進行疊加
    vessel_overlay = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    vessel_overlay[vessel_mask == 1] = [255, 0, 0]  # 將血管區域標記為紅色

    # 計算512x512圖片中視盤中心點的相對位置
    scale_x = 512 / original_image.shape[1]
    scale_y = 512 / original_image.shape[0]
    od_center_512 = (int(od_center[0] * scale_x), int(od_center[1] * scale_y))

    # 根據視盤中心點切割象限
    height, width = vessel_overlay.shape[:2]
    quadrants = {
        'Q1': vessel_overlay[0:od_center_512[1], od_center_512[0]:width],
        'Q2': vessel_overlay[0:od_center_512[1], 0:od_center_512[0]],
        'Q3': vessel_overlay[od_center_512[1]:height, 0:od_center_512[0]],
        'Q4': vessel_overlay[od_center_512[1]:height, od_center_512[0]:width]
    }

    # 保存結果
    save_quadrant_images(quadrants, image_path)

    # 顯示結果
    plt.figure(figsize=(15, 10))
    
    # 顯示原始圖片與ZONE1標記
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_with_zone1, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with ZONE1')

    # 顯示血管分割結果
    plt.subplot(2, 3, 2)
    plt.imshow(vessel_overlay)
    plt.title('Vessel Segmentation')
    
    # 顯示四個象限
    plt.subplot(2, 3, 3)
    plt.imshow(quadrants['Q1'])
    plt.title('Quadrant 1')
    
    plt.subplot(2, 3, 4)
    plt.imshow(quadrants['Q2'])
    plt.title('Quadrant 2')
    
    plt.subplot(2, 3, 5)
    plt.imshow(quadrants['Q3'])
    plt.title('Quadrant 3')
    
    plt.subplot(2, 3, 6)
    plt.imshow(quadrants['Q4'])
    plt.title('Quadrant 4')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
