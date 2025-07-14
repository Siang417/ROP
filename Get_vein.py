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

def create_improved_circular_mask(image):
    """
    改進的圓形遮罩建立方法
    使用多重策略確保更準確的眼球區域檢測
    """
    # 轉換為灰階
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # 方法1: 基於亮度的初始遮罩
    # 使用較低的閾值來捕捉更多的眼球區域
    mean_intensity = np.mean(gray)
    threshold = max(mean_intensity * 0.1, 10)  # 動態閾值
    
    _, mask1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 方法2: 基於邊緣的圓形檢測
    # 使用霍夫圓變換檢測眼球邊界
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=min(h, w)//2,
        param1=50, 
        param2=30, 
        minRadius=min(h, w)//4, 
        maxRadius=min(h, w)//2
    )
    
    mask2 = np.zeros_like(gray)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 選擇最大的圓
        largest_circle = max(circles, key=lambda x: x[2])
        cv2.circle(mask2, (largest_circle[0], largest_circle[1]), largest_circle[2], 255, -1)
    
    # 方法3: 基於連通區域的方法
    # 找到最大的連通區域
    mask3 = mask1.copy()
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到最大輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        mask3 = np.zeros_like(gray)
        cv2.fillPoly(mask3, [largest_contour], 255)
        
        # 使用凸包來平滑邊界
        hull = cv2.convexHull(largest_contour)
        cv2.fillPoly(mask3, [hull], 255)
    
    # 結合三種方法
    # 如果霍夫圓檢測成功，優先使用；否則使用連通區域方法
    if np.sum(mask2) > 0:
        final_mask = mask2
        method_used = "霍夫圓檢測"
    else:
        final_mask = mask3
        method_used = "連通區域檢測"
    
    # 形態學操作來平滑遮罩
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # 使用中值濾波進一步平滑
    final_mask = cv2.medianBlur(final_mask, 5)
    
    print(f"遮罩建立方法: {method_used}")
    return final_mask

def analyze_all_channels(image, mask):
    """
    分析所有RGB通道，選擇最佳通道用於血管檢測
    """
    if len(image.shape) == 3:
        r_channel = image[:,:,0]
        g_channel = image[:,:,1] 
        b_channel = image[:,:,2]
    else:
        return image, "灰階", {}  # 已經是灰階
    
    # 計算每個通道的血管檢測適用性指標
    def calculate_vessel_metrics(channel, mask):
        """計算血管檢測相關指標"""
        # 應用遮罩
        masked_channel = cv2.bitwise_and(channel, mask)
        masked_pixels = masked_channel[mask > 0]
        
        if len(masked_pixels) == 0:
            return 0, 0, 0
        
        # 1. 對比度 (標準差)
        contrast = np.std(masked_pixels)
        
        # 2. 血管清晰度 (基於邊緣強度)
        sobelx = cv2.Sobel(masked_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(masked_channel, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        edge_strength = np.mean(sobel_combined[mask > 0])
        
        # 3. 動態範圍
        dynamic_range = np.max(masked_pixels) - np.min(masked_pixels)
        
        return contrast, edge_strength, dynamic_range
    
    # 分析各通道
    channels = {'紅色': r_channel, '綠色': g_channel, '藍色': b_channel}
    results = {}
    
    for name, channel in channels.items():
        contrast, edge_strength, dynamic_range = calculate_vessel_metrics(channel, mask)
        
        # 綜合評分 (權重可調整)
        score = (contrast * 0.3 + edge_strength * 0.5 + dynamic_range * 0.2)
        
        results[name] = {
            'contrast': contrast,
            'edge_strength': edge_strength,
            'dynamic_range': dynamic_range,
            'score': score,
            'channel': channel
        }
    
    # 選擇最佳通道
    best_channel_name = max(results.keys(), key=lambda x: results[x]['score'])
    best_channel = results[best_channel_name]['channel']
    
    return best_channel, best_channel_name, results

def sobel_vessel_detection(image, mask, selected_channel):
    """
    使用Sobel邊緣檢測方法進行改良的血管檢測
    輸入: 原始影像 + 遮罩 + 選定的通道
    """
    # 使用選定的通道
    if selected_channel is not None:
        gray = selected_channel.copy()
    else:
        # 如果沒有選定通道，轉換為灰階
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
    
    # 步驟7：形態學操作
    kernel = np.ones((3, 3), np.uint8)
    
    for i in range(3):
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
    
    # 對最終結果應用原始遮罩
    binary = cv2.bitwise_and(binary, mask)
    
    return binary, sobel_combined, sobel_processed, sobel_eq, threshold_value

def channel_vessel_detection(image, mask, selected_channel):
    """
    基於選定通道的血管檢測方法
    輸入: 原始影像 + 遮罩 + 選定的通道
    """
    if selected_channel is not None:
        channel = selected_channel.copy()
    else:
        if len(image.shape) == 3:
            channel = image[:, :, 1]  # 預設使用綠色通道
        else:
            channel = image
    
    # 應用遮罩
    channel = channel * (mask / 255.0)
    
    # 應用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(channel.astype(np.uint8))
    
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
    主要處理函數，使用改良的血管檢測方法
    """
    print("🔬 增強型眼底血管分割系統 v2.0")
    print("=" * 50)
    
    # 設定初始目錄
    initial_dir = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Quadrant_division"
    
    # 選擇影像檔案
    print("請選擇眼底影像檔案...")
    image_path = select_image_file(initial_dir)
    
    if not image_path:
        print("未選擇檔案，程式結束。")
        return
    
    # 載入並調整影像大小
    try:
        image = np.array(Image.open(image_path))
        print(f"✅ 成功載入影像：{os.path.basename(image_path)}")
        print(f"📏 原始影像大小：{image.shape[1]} x {image.shape[0]} 像素")
        
        # 調整影像大小
        image = resize_image(image, max_size=640)
        print(f"📏 調整後影像大小：{image.shape[1]} x {image.shape[0]} 像素")
        
    except Exception as e:
        print(f"❌ 載入影像失敗：{e}")
        return
    
    print("\n🔄 處理步驟：")
    print("-" * 30)
    
    # 步驟1：建立改進的圓形遮罩
    print("步驟1：建立改進的圓形遮罩...")
    mask = create_improved_circular_mask(image)
    
    # 步驟2：自適應通道選擇
    print("步驟2：分析並選擇最佳通道...")
    best_channel, best_channel_name, channel_analysis = analyze_all_channels(image, mask)
    
    print(f"📊 通道分析結果：")
    for name, metrics in channel_analysis.items():
        print(f"  {name:4}: 對比度={metrics['contrast']:.2f}, 邊緣強度={metrics['edge_strength']:.2f}, 動態範圍={metrics['dynamic_range']:.2f}, 總分={metrics['score']:.2f}")
    print(f"🏆 選擇通道: {best_channel_name}")
    
    # 步驟3：Sobel血管檢測 (使用原始影像 + 遮罩 + 最佳通道)
    print("步驟3：應用改良的Sobel血管檢測...")
    sobel_vessels, sobel_raw, sobel_processed, sobel_eq, threshold_val = sobel_vessel_detection(image, mask, best_channel)
    
    # 步驟4：通道血管檢測 (使用原始影像 + 遮罩 + 最佳通道)
    print("步驟4：應用基於最佳通道的血管檢測...")
    channel_vessels, channel_enhanced, channel_threshold = channel_vessel_detection(image, mask, best_channel)
    
    # 步驟5：建立視覺化
    print("步驟5：建立視覺化...")
    
    # 建立覆蓋層
    sobel_overlay = np.zeros_like(image)
    sobel_overlay[sobel_vessels > 0] = [255, 0, 0]  # Sobel血管用紅色
    
    channel_overlay = np.zeros_like(image)
    channel_overlay[channel_vessels > 0] = [0, 255, 0]  # 通道血管用綠色
    
    # 與原始影像混合
    sobel_result = (image * 0.7 + sobel_overlay * 0.3).astype(np.uint8)
    channel_result = (image * 0.7 + channel_overlay * 0.3).astype(np.uint8)
    
    # 對結果應用遮罩
    for i in range(3):
        sobel_result[:, :, i] = sobel_result[:, :, i] * (mask / 255.0)
        channel_result[:, :, i] = channel_result[:, :, i] * (mask / 255.0)
    
    # 顯示RGB通道分析
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
    
    if len(image.shape) == 3:
        axes1[0].imshow(image[:,:,0], cmap='Reds')
        axes1[0].set_title('紅色通道', fontsize=12)
        axes1[0].axis('off')
        
        axes1[1].imshow(image[:,:,1], cmap='Greens') 
        axes1[1].set_title('綠色通道', fontsize=12)
        axes1[1].axis('off')
        
        axes1[2].imshow(image[:,:,2], cmap='Blues')
        axes1[2].set_title('藍色通道', fontsize=12)
        axes1[2].axis('off')
        
        axes1[3].imshow(best_channel, cmap='gray')
        axes1[3].set_title(f'選定的{best_channel_name}通道', fontsize=12)
        axes1[3].axis('off')
    
    plt.suptitle('RGB通道分析與最佳通道選擇', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 顯示主要處理結果
    fig2, axes2 = plt.subplots(3, 4, figsize=(20, 15))
    
    # 第一行：基礎處理步驟
    axes2[0, 0].imshow(image)
    axes2[0, 0].set_title('原始眼底影像', fontsize=11)
    axes2[0, 0].axis('off')
    
    axes2[0, 1].imshow(mask, cmap='gray')
    axes2[0, 1].set_title('改進的圓形遮罩', fontsize=11)
    axes2[0, 1].axis('off')
    
    axes2[0, 2].imshow(sobel_raw, cmap='gray')
    axes2[0, 2].set_title('Sobel邊緣檢測（原始）', fontsize=11)
    axes2[0, 2].axis('off')
    
    axes2[0, 3].imshow(sobel_processed, cmap='gray')
    axes2[0, 3].set_title('邊緣抑制後及統計移除', fontsize=11)
    axes2[0, 3].axis('off')
    
    # 第二行：Sobel方法結果
    axes2[1, 0].imshow(sobel_eq, cmap='gray')
    axes2[1, 0].set_title('直方圖均衡化', fontsize=11)
    axes2[1, 0].axis('off')
    
    axes2[1, 1].imshow(sobel_vessels, cmap='gray')
    axes2[1, 1].set_title(f'Sobel血管遮罩（閾值：{threshold_val}）', fontsize=11)
    axes2[1, 1].axis('off')
    
    axes2[1, 2].imshow(sobel_overlay)
    axes2[1, 2].set_title('Sobel血管覆蓋（紅色）', fontsize=11)
    axes2[1, 2].axis('off')
    
    axes2[1, 3].imshow(sobel_result)
    axes2[1, 3].set_title('Sobel方法最終結果', fontsize=11)
    axes2[1, 3].axis('off')
    
    # 第三行：最佳通道方法結果
    axes2[2, 0].imshow(channel_enhanced, cmap='gray')
    axes2[2, 0].set_title(f'{best_channel_name}通道增強', fontsize=11)
    axes2[2, 0].axis('off')
    
    axes2[2, 1].imshow(channel_vessels, cmap='gray')
    axes2[2, 1].set_title(f'{best_channel_name}通道血管（閾值：{channel_threshold:.1f}）', fontsize=11)
    axes2[2, 1].axis('off')
    
    axes2[2, 2].imshow(channel_overlay)
    axes2[2, 2].set_title(f'{best_channel_name}通道覆蓋（綠色）', fontsize=11)
    axes2[2, 2].axis('off')
    
    axes2[2, 3].imshow(channel_result)
    axes2[2, 3].set_title(f'{best_channel_name}通道方法最終結果', fontsize=11)
    axes2[2, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'增強型眼底血管分割：Sobel vs {best_channel_name}通道方法', 
                 fontsize=16, y=0.98)
    plt.show()
    
    # 列印詳細分析結果
    print("\n📊 處理摘要：")
    print("-" * 30)
    print(f"• 影像大小：{image.shape[1]} x {image.shape[0]} 像素")
    print(f"• 選擇的最佳通道：{best_channel_name}")
    print(f"• 遮罩覆蓋的眼球區域：{np.sum(mask > 0):,} 像素")
    print(f"• Sobel統計閾值：{threshold_val}")
    print(f"• Sobel檢測到的血管像素：{np.sum(sobel_vessels > 0):,}")
    print(f"• {best_channel_name}通道閾值：{channel_threshold:.2f}")
    print(f"• {best_channel_name}通道檢測到的血管像素：{np.sum(channel_vessels > 0):,}")
    print(f"• Sobel血管覆蓋率：{(np.sum(sobel_vessels > 0) / np.sum(mask > 0)) * 100:.2f}%")
    print(f"• {best_channel_name}通道血管覆蓋率：{(np.sum(channel_vessels > 0) / np.sum(mask > 0)) * 100:.2f}%")
    
    print("\n🎯 通道分析詳細結果：")
    for name, metrics in channel_analysis.items():
        print(f"  {name}通道 - 對比度: {metrics['contrast']:.2f}, 邊緣強度: {metrics['edge_strength']:.2f}, 動態範圍: {metrics['dynamic_range']:.2f}")
    
    return {
        'original_image': image,
        'mask': mask,
        'best_channel': best_channel,
        'best_channel_name': best_channel_name,
        'channel_analysis': channel_analysis,
        'sobel_vessels': sobel_vessels,
        'channel_vessels': channel_vessels,
        'sobel_result': sobel_result,
        'channel_result': channel_result
    }

# 主程式執行
if __name__ == "__main__":
    result = process_retinal_image()