import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import filedialog, Tk
import os
import cv2
from scipy import ndimage
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def select_image_file(initial_dir):
    """
    開啟檔案對話框選擇影像檔案
    """
    root = Tk()
    root.withdraw()  # 隱藏主視窗
    
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="選擇眼底影像",
        filetypes=[("影像檔案", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
    )
    
    root.destroy()
    return file_path

def resize_image(image, max_size=640):
    """
    調整影像大小以適應最大尺寸，同時保持長寬比
    """
    if len(image.shape) == 3:
        h, w = image.shape[:2]
    else:
        h, w = image.shape
    
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    return image

def create_circular_mask(image):
    """
    建立圓形遮罩以排除眼球外的黑色背景
    """
    # 如需要則轉換為灰階
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 應用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用閾值分離眼球與背景
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 應用形態學操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 找到最大輪廓（眼球）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    return mask

def sobel_vessel_detection(image, mask):
    """
    使用Sobel邊緣檢測方法進行改良的血管檢測
    """
    # 如需要則轉換為灰階
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 應用遮罩
    gray = cv2.bitwise_and(gray, mask)
    
    # 步驟1：中值濾波減少噪聲
    img_median = cv2.medianBlur(gray, 3)
    
    # 步驟2：Sobel邊緣檢測
    sobel_x = cv2.Sobel(img_median, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    
    sobel_y = cv2.Sobel(img_median, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # 結合X + Y邊緣
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    
    # 步驟3：抑制高強度邊緣（移除視神經盤等強邊緣）
    sobel_processed = sobel_combined.copy()
    coords = np.column_stack(np.where(sobel_processed >= 120))
    r = 3  # 抑制半徑
    
    for y, x in coords:
        x1 = max(x - r, 0)
        x2 = min(x + r + 1, sobel_processed.shape[1])
        y1 = max(y - r, 0)
        y2 = min(y + r + 1, sobel_processed.shape[0])
        sobel_processed[y1:y2, x1:x2] = 0
    
    # 步驟4：統計像素移除（移除最低50%像素值）
    pixel_counts = np.bincount(sobel_processed.flatten(), minlength=256)
    total_pixels = sobel_processed.size
    pixels_to_remove = int(total_pixels * 0.5)
    
    cumulative_count = 0
    threshold_value = 0
    
    for pixel_value in range(256):
        cumulative_count += pixel_counts[pixel_value]
        if cumulative_count >= pixels_to_remove:
            threshold_value = pixel_value
            break
    
    sobel_processed[sobel_processed <= threshold_value] = 0
    
    # 步驟5：對5-255範圍內的像素進行直方圖均衡化
    mask_eq = (sobel_processed >= 5) & (sobel_processed <= 255)
    sobel_eq = sobel_processed.copy()
    
    if np.any(mask_eq):
        roi = sobel_processed[mask_eq]
        roi_eq = cv2.equalizeHist(roi.reshape(-1, 1).astype(np.uint8)).flatten()
        sobel_eq[mask_eq] = roi_eq
    
    # 步驟6：二值化閾值處理（保留像素值 >= 250）
    binary = np.zeros_like(sobel_eq)
    binary[sobel_eq >= 250] = 255
    
    # 步驟7：形態學操作（5次膨脹+侵蝕迭代）
    kernel = np.ones((3, 3), np.uint8)  # 較小的核心以更好地保持血管
    
    for i in range(3):  # 減少迭代次數
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
    
    # 對最終結果應用原始遮罩
    binary = cv2.bitwise_and(binary, mask)
    
    return binary, sobel_combined, sobel_processed, sobel_eq, threshold_value

def green_channel_vessel_detection(image, mask):
    """
    傳統綠色通道血管檢測用於比較
    """
    if len(image.shape) == 3:
        green_channel = image[:, :, 1]  # 綠色通道
    else:
        green_channel = image
    
    # 應用遮罩
    green_channel = green_channel * (mask / 255.0)
    
    # 應用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel.astype(np.uint8))
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 自適應閾值
    mask_bool = mask > 0
    if np.sum(mask_bool) > 0:
        mean_val = np.mean(blurred[mask_bool])
        std_val = np.std(blurred[mask_bool])
        threshold_val = max(mean_val - 1.0 * std_val, 0)
    else:
        threshold_val = 100
    
    vessel_mask = (blurred < threshold_val) & (mask > 0)
    vessel_mask = vessel_mask.astype(np.uint8) * 255
    
    # 形態學操作
    vessel_mask = morphology.binary_opening(vessel_mask, morphology.disk(1))
    vessel_mask = morphology.binary_closing(vessel_mask, morphology.disk(2))
    vessel_mask = vessel_mask.astype(np.uint8) * 255
    
    return vessel_mask, enhanced, threshold_val

def process_retinal_image():
    """
    主要處理函數，使用改良的Sobel方法
    """
    print("增強型眼底血管分割系統")
    print("=" * 50)
    
    # 設定初始目錄
    initial_dir = r"C:\\Users\\Zz423\\Desktop\\研究所\\UCL\\旺宏\\Redina 資料\\Quadrant_division"
    
    # 選擇影像檔案
    print("請選擇眼底影像檔案...")
    image_path = select_image_file(initial_dir)
    
    if not image_path:
        print("未選擇檔案，程式結束。")
        return
    
    # 載入並調整影像大小
    try:
        image = np.array(Image.open(image_path))
        print(f"成功載入影像：{os.path.basename(image_path)}")
        print(f"原始影像大小：{image.shape[1]} x {image.shape[0]} 像素")
        
        # 調整影像大小
        image = resize_image(image, max_size=640)
        print(f"調整後影像大小：{image.shape[1]} x {image.shape[0]} 像素")
        
    except Exception as e:
        print(f"載入影像失敗：{e}")
        return
    
    print("\\n處理步驟：")
    print("-" * 30)
    
    # 步驟1：建立圓形遮罩
    print("步驟1：建立圓形遮罩...")
    mask = create_circular_mask(image)
    
    # 步驟2：Sobel血管檢測
    print("步驟2：應用改良的Sobel血管檢測...")
    sobel_vessels, sobel_raw, sobel_processed, sobel_eq, threshold_val = sobel_vessel_detection(image, mask)
    
    # 步驟3：綠色通道檢測用於比較
    print("步驟3：應用傳統綠色通道檢測...")
    green_vessels, green_enhanced, green_threshold = green_channel_vessel_detection(image, mask)
    
    # 步驟4：建立視覺化
    print("步驟4：建立視覺化...")
    
    # 建立覆蓋層
    sobel_overlay = np.zeros_like(image)
    sobel_overlay[sobel_vessels > 0] = [255, 0, 0]  # Sobel血管用紅色
    
    green_overlay = np.zeros_like(image)
    green_overlay[green_vessels > 0] = [0, 255, 0]  # 傳統血管用綠色
    
    # 與原始影像混合
    sobel_result = (image * 0.7 + sobel_overlay * 0.3).astype(np.uint8)
    green_result = (image * 0.7 + green_overlay * 0.3).astype(np.uint8)
    
    # 對結果應用遮罩
    for i in range(3):
        sobel_result[:, :, i] = sobel_result[:, :, i] * (mask / 255.0)
        green_result[:, :, i] = green_result[:, :, i] * (mask / 255.0)
    
    # 顯示結果
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 第一行：原始處理步驟
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始眼底影像', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('圓形遮罩', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sobel_raw, cmap='gray')
    axes[0, 2].set_title('Sobel邊緣檢測（原始）', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(sobel_processed, cmap='gray')
    axes[0, 3].set_title('邊緣抑制後及統計移除', fontsize=11)
    axes[0, 3].axis('off')
    
    # 第二行：Sobel方法結果
    axes[1, 0].imshow(sobel_eq, cmap='gray')
    axes[1, 0].set_title('直方圖均衡化', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sobel_vessels, cmap='gray')
    axes[1, 1].set_title(f'Sobel血管遮罩（閾值：{threshold_val}）', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sobel_overlay)
    axes[1, 2].set_title('Sobel血管覆蓋（紅色）', fontsize=11)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(sobel_result)
    axes[1, 3].set_title('Sobel方法結果', fontsize=11)
    axes[1, 3].axis('off')
    
    # 第三行：綠色通道方法用於比較
    axes[2, 0].imshow(green_enhanced, cmap='gray')
    axes[2, 0].set_title('綠色通道增強', fontsize=11)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(green_vessels, cmap='gray')
    axes[2, 1].set_title(f'綠色通道血管（閾值：{green_threshold:.1f}）', fontsize=11)
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(green_overlay)
    axes[2, 2].set_title('綠色通道覆蓋（綠色）', fontsize=11)
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(green_result)
    axes[2, 3].set_title('綠色通道結果', fontsize=11)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle('增強型眼底血管分割：Sobel vs 綠色通道方法', 
                 fontsize=16, y=0.98)
    
    # 列印分析結果
    print("\\n處理摘要：")
    print("-" * 30)
    print(f"• 影像大小：{image.shape[1]} x {image.shape[0]} 像素")
    print(f"• Sobel統計閾值：{threshold_val}")
    print(f"• Sobel檢測到的血管像素：{np.sum(sobel_vessels > 0):,}")
    print(f"• 綠色通道閾值：{green_threshold:.2f}")
    print(f"• 綠色通道檢測到的血管像素：{np.sum(green_vessels > 0):,}")
    print(f"• 眼球區域：{np.sum(mask > 0):,} 像素")
    print(f"• Sobel血管覆蓋率：{(np.sum(sobel_vessels > 0) / np.sum(mask > 0)) * 100:.2f}%")
    print(f"• 綠色通道血管覆蓋率：{(np.sum(green_vessels > 0) / np.sum(mask > 0)) * 100:.2f}%")
    
    plt.show()
    
    return {
        'original_image': image,
        'mask': mask,
        'sobel_vessels': sobel_vessels,
        'green_vessels': green_vessels,
        'sobel_result': sobel_result,
        'green_result': green_result
    }

# 主程式執行
if __name__ == "__main__":
    results = process_retinal_image()