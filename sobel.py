import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# 創建 tkinter 根窗口（隱藏）
root = tk.Tk()
root.withdraw()

# 設置初始目錄
initial_dir = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Quadrant_division"

# 檢查目錄是否存在，如果不存在則使用當前目錄
if not os.path.exists(initial_dir):
    initial_dir = os.getcwd()
    print(f"指定目錄不存在，使用當前目錄: {initial_dir}")

# 打開文件選擇對話框
file_path = filedialog.askopenfilename(
    title="選擇影像文件",
    initialdir=initial_dir,
    filetypes=[
        ("影像文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
        ("PNG文件", "*.png"),
        ("JPEG文件", "*.jpg *.jpeg"),
        ("所有文件", "*.*")
    ]
)

# 檢查是否選擇了文件
if not file_path:
    print("未選擇文件，程序退出")
    exit()

print(f"選擇的文件: {file_path}")

# 讀取原始圖像
img_original = cv2.imread(file_path, cv2.IMREAD_COLOR)
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# 檢查圖像是否成功載入
if img_original is None or img is None:
    print("無法載入圖像，請檢查文件格式")
    exit()


# 等比例縮小 img，長或寬不能超過 640
if img is not None:
    h, w = img.shape[:2]
    scale = min(640 / h, 640 / w, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_original = cv2.resize(img_original, (new_w, new_h), interpolation=cv2.INTER_AREA)

cv2.imshow('Original', img)

# 對 img 做中值濾波器
img_median = cv2.medianBlur(img, 3)  # 5x5 的中值濾波器
cv2.imshow('Sobel Median Filter', img_median)

# 更新 sobel_combined_2 為濾波後的結果
img = img_median





# 計算 X 軸（垂直邊緣）
sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# 計算 Y 軸（水平方向邊緣）
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

# 合併 X + Y 邊緣
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

sobel_combined_2 = sobel_combined.copy()


# 找出值 ≥ 200 的座標
coords = np.column_stack(np.where(sobel_combined_2 >= 120))
# 定義半徑（5x5 區域 → 半徑為 2）
r = 3
# 針對每個符合條件的像素，清除它周圍的 5x5 區域
for y, x in coords:
    x1 = max(x - r, 0)
    x2 = min(x + r + 1, img.shape[1])
    y1 = max(y - r, 0)
    y2 = min(y + r + 1, img.shape[0])
    sobel_combined_2[y1:y2, x1:x2] = 0
    
cv2.imshow('Sobel Combined_2', sobel_combined_2)

# 計算0-255像素值的數量並刪除總數的20%
# 統計每個像素值的數量
pixel_counts = np.bincount(sobel_combined_2.flatten(), minlength=256)
total_pixels = sobel_combined_2.size
pixels_to_remove = int(total_pixels * 0.5)  # 計算要刪除的像素數量(20%)

print(f"總像素數: {total_pixels}")
print(f"要刪除的像素數: {pixels_to_remove}")

# 從低像素值開始累積，找出要刪除的像素值範圍
cumulative_count = 0
threshold_value = 0

for pixel_value in range(256):
    cumulative_count += pixel_counts[pixel_value]
    if cumulative_count >= pixels_to_remove:
        threshold_value = pixel_value
        break

print(f"刪除閾值: {threshold_value} (像素值 0-{threshold_value} 將被設為 0)")

# 將低於閾值的像素設為0
sobel_combined_2[sobel_combined_2 <= threshold_value] = 0

cv2.imshow('Sobel Combined_2 After Removal', sobel_combined_2)

# 凸顯亮細節的處理
# 方法1: Histogram Equalization - 只對 30-255 像素值做增強
# 創建遮罩，只選擇像素值在 30-255 範圍內的像素
mask = (sobel_combined_2 >= 5) & (sobel_combined_2 <= 255)

# 複製原圖像
sobel_eq = sobel_combined_2.copy()

# 只對遮罩範圍內的像素做直方圖均衡化
if np.any(mask):
    # 提取感興趣區域
    roi = sobel_combined_2[mask]
    # 對提取的像素做直方圖均衡化
    roi_eq = cv2.equalizeHist(roi.reshape(-1, 1).astype(np.uint8)).flatten()
    # 將均衡化結果放回原位置
    sobel_eq[mask] = roi_eq

# # 方法2: CLAHE (Contrast Limited Adaptive Histogram Equalization) - 局部對比度增強
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# sobel_clahe = clahe.apply(sobel_combined_2)

# # 選擇最佳效果用於後續處理 (這裡使用CLAHE)
# sobel_combined_2 = sobel_clahe

# # 方法3: Gamma校正 - 提亮暗部細節 (gamma < 1 會提亮)
# gamma = 0.7
# sobel_gamma = np.power(sobel_combined_2 / 255.0, gamma) * 255.0
# sobel_gamma = np.uint8(sobel_gamma)

 

# 顯示結果比較
cv2.imshow('Histogram Equalized', sobel_eq)
# cv2.imshow('CLAHE Enhanced', sobel_clahe)
# cv2.imshow('Gamma Corrected', sobel_gamma)



# 建立空白圖（全黑）
binary = np.zeros_like(sobel_eq)

# 條件遮罩：只保留像素值在 130~150 之間的像素
binary[(sobel_eq >= 250)] = 255

cv2.imshow('binary', binary)
# cv2.imwrite('binary.png', binary)

# 形態學操作：先侵蝕再膨脹
# 這樣可以去除小的噪點，並平滑邊
kernel = np.ones((5, 5), np.uint8)  # 定義3x3的結構元素

for i in range(5):  # 重複兩次侵蝕和膨脹
    binary = cv2.dilate(binary, kernel, iterations=1)  # 再膨脹
    binary = cv2.erode(binary, kernel, iterations=1)    # 先侵蝕

cv2.imshow('binary_dilated', binary)
# cv2.imwrite('binary_dilated.png', binary)

# 連通元件分析
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)  # 找最大面積的輪廓
area = cv2.contourArea(max_contour)               # 最大面積

# 創建最大連通元件的遮罩
mask = np.zeros(binary.shape, dtype=np.uint8)
cv2.fillPoly(mask, [max_contour], 255)

# 獲取連通元件內的像素位置
component_pixels = np.where(mask == 255)
component_coords = list(zip(component_pixels[0], component_pixels[1]))

print(f"連通元件包含 {len(component_coords)} 個像素")

# 基於區域成長的方法：使用原始影像強度進行擴展
def region_growing(original_img, seed_coords, intensity_threshold=15):
    """
    基於影像強度的區域成長算法
    """
    h, w = original_img.shape
    visited = np.zeros((h, w), dtype=bool)
    result_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 計算種子區域的平均強度
    seed_intensities = [original_img[y, x] for y, x in seed_coords]
    mean_intensity = np.mean(seed_intensities)
    
    print(f"種子區域平均強度: {mean_intensity:.2f}")
    
    # 使用BFS進行區域成長
    from collections import deque
    queue = deque(seed_coords)
    
    # 標記種子點
    for y, x in seed_coords:
        visited[y, x] = True
        result_mask[y, x] = 255
    
    # 8-連通的鄰域
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        current_y, current_x = queue.popleft()
        current_intensity = original_img[current_y, current_x]
        
        # 檢查8個鄰域
        for dy, dx in directions:
            ny, nx = current_y + dy, current_x + dx
            
            # 檢查邊界
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                neighbor_intensity = original_img[ny, nx]
                
                # 檢查強度差異
                intensity_diff = abs(float(neighbor_intensity) - float(mean_intensity))
                
                if intensity_diff <= intensity_threshold:
                    visited[ny, nx] = True
                    result_mask[ny, nx] = 255
                    queue.append((ny, nx))
    
    return result_mask

# 應用區域成長到原始灰階影像
print("開始基於影像強度的區域成長...")
grown_mask = region_growing(img, component_coords, intensity_threshold=20)

print(f"區域成長後包含 {np.sum(grown_mask == 255)} 個像素")
print(f"擴展比例: {np.sum(grown_mask == 255) / len(component_coords):.2f}x")

# 可選：對成長後的結果進行形態學平滑
kernel_smooth = np.ones((3, 3), np.uint8)
grown_mask_smoothed = cv2.morphologyEx(grown_mask, cv2.MORPH_CLOSE, kernel_smooth)
grown_mask_smoothed = cv2.morphologyEx(grown_mask_smoothed, cv2.MORPH_OPEN, kernel_smooth)

# 顯示結果比較
cv2.imshow('Original Connected Component', mask)
cv2.imshow('Region Growing Result', grown_mask)
cv2.imshow('Region Growing Smoothed', grown_mask_smoothed)

# 在原圖上顯示結果
img_color = img_original.copy()

# 繪製原始連通元件（紅色）
cv2.drawContours(img_color, [max_contour], -1, (0, 0, 255), 2)

# 繪製區域成長結果（綠色）
grown_contours, _ = cv2.findContours(grown_mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if grown_contours:
    largest_grown = max(grown_contours, key=cv2.contourArea)
    cv2.drawContours(img_color, [largest_grown], -1, (0, 255, 0), 2)

cv2.imshow('Comparison: Red=Original, Green=Region Growing', img_color)

# 保存結果
# cv2.imwrite('original_component.png', mask)
# cv2.imwrite('region_growing_result.png', grown_mask_smoothed)
# cv2.imwrite('comparison_result.png', img_color)



# 顯示
# cv2.imshow('Original', img_original)
# cv2.imshow('Sobel X', sobel_x)
# cv2.imshow('Sobel Y', sobel_y)
# cv2.imshow('Sobel Combined', sobel_combined)
# cv2.imwrite('sobel_combined.png', sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
