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
    改進版Sobel邊緣檢測方法進行血管檢測
    改進：統一使用最佳通道，調整處理順序
    """
    # 使用選定的最佳通道
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
    
    # 步驟3：先進行直方圖均衡化（改進：提前增強對比度）
    sobel_eq = cv2.equalizeHist(sobel_combined)
    
    # 步驟4：抑制高強度邊緣（移除視神經盤等強邊緣）
    sobel_processed = sobel_eq.copy()
    coords = np.column_stack(np.where(sobel_processed >= 120))
    r = 3  # 抑制半徑
    
    for y, x in coords:
        x1 = max(x - r, 0)
        x2 = min(x + r + 1, sobel_processed.shape[1])
        y1 = max(y - r, 0)
        y2 = min(y + r + 1, sobel_processed.shape[0])
        sobel_processed[y1:y2, x1:x2] = 0
    
    # 步驟5：統計像素移除（移除最低50%像素值）
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
    
    # 步驟6：二值化閾值處理（保留像素值 >= 250）
    binary = np.zeros_like(sobel_processed)
    binary[sobel_processed >= 250] = 255
    
    # 步驟7：形態學操作
    kernel = np.ones((3, 3), np.uint8)
    
    for i in range(3):
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
    
    # 對最終結果應用原始遮罩
    binary = cv2.bitwise_and(binary, mask)
    
    return binary, sobel_combined, sobel_processed, sobel_eq, threshold_value

def channel_vessel_detection(image, mask, selected_channel, k_value=1.5):
    """
    改進版基於選定通道的血管檢測方法
    改進：使用k=1.5提高精確度，調整處理順序
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
    
    # 步驟1：應用CLAHE增強對比度（改進：提前增強）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channel_enhanced = clahe.apply(channel.astype(np.uint8))
    
    # 步驟2：邊緣檢測
    sobel_x = cv2.Sobel(channel_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(channel_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = np.uint8(edges / edges.max() * 255)
    
    # 步驟3：邊緣抑制（移除強邊緣）
    edges_suppressed = edges.copy()
    coords = np.column_stack(np.where(edges >= 120))
    r = 3
    
    for y, x in coords:
        x1 = max(x - r, 0)
        x2 = min(x + r + 1, edges_suppressed.shape[1])
        y1 = max(y - r, 0)
        y2 = min(y + r + 1, edges_suppressed.shape[0])
        edges_suppressed[y1:y2, x1:x2] = 0
    
    # 步驟4：統計移除（移除最低50%）
    pixel_counts = np.bincount(edges_suppressed.flatten(), minlength=256)
    total_pixels = edges_suppressed.size
    pixels_to_remove = int(total_pixels * 0.5)
    
    cumulative_count = 0
    statistical_threshold = 0
    
    for pixel_value in range(256):
        cumulative_count += pixel_counts[pixel_value]
        if cumulative_count >= pixels_to_remove:
            statistical_threshold = pixel_value
            break
    
    edges_suppressed[edges_suppressed <= statistical_threshold] = 0
    
    # 步驟5：再次直方圖均衡化（進一步增強）
    if np.max(edges_suppressed) > 0:
        edges_final = cv2.equalizeHist(edges_suppressed)
    else:
        edges_final = edges_suppressed
    
    # 步驟6：自適應閾值（使用k=1.5）
    masked_pixels = channel_enhanced[mask > 0]
    if len(masked_pixels) > 0:
        mean_val = np.mean(masked_pixels)
        std_val = np.std(masked_pixels)
        adaptive_threshold = mean_val - k_value * std_val  # 使用k=1.5
        
        print(f"自適應閾值參數: 平均值={mean_val:.2f}, 標準差={std_val:.2f}")
        print(f"計算閾值 (k={k_value}): {adaptive_threshold:.2f}")
    else:
        adaptive_threshold = 128
    
    # 二值化
    binary = np.zeros_like(channel_enhanced)
    binary[channel_enhanced < adaptive_threshold] = 255
    binary = cv2.bitwise_and(binary, mask)
    
    # 形態學操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary, channel_enhanced, edges_final, adaptive_threshold

def calculate_performance_metrics(binary_result, mask):
    """
    計算血管檢測性能指標
    """
    # 計算檢測到的血管像素數量
    vessel_pixels = np.sum(binary_result == 255)
    total_roi_pixels = np.sum(mask == 255)
    
    # 血管覆蓋率
    vessel_coverage = (vessel_pixels / total_roi_pixels * 100) if total_roi_pixels > 0 else 0
    
    # 連通區域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_result, connectivity=8)
    
    # 過濾小區域（面積 < 10像素）
    large_components = stats[stats[:, cv2.CC_STAT_AREA] >= 10]
    num_vessels = len(large_components) - 1  # 減去背景
    
    metrics = {
        'vessel_pixels': vessel_pixels,
        'vessel_coverage': vessel_coverage,
        'num_vessels': max(0, num_vessels),
        'total_roi_pixels': total_roi_pixels
    }
    
    return metrics

def display_results(original, mask, best_channel, best_channel_name, channel_results, 
                   sobel_results, channel_detection_results, sobel_metrics, channel_metrics):
    """
    顯示改進版檢測結果
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 第一行：原始影像和通道分析
    plt.subplot(4, 5, 1)
    plt.imshow(original)
    plt.title('原始眼底影像', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('眼球區域遮罩', fontsize=12)
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(best_channel, cmap='gray')
    plt.title(f'最佳通道: {best_channel_name}', fontsize=12, fontweight='bold', color='red')
    plt.axis('off')
    
    # 通道分析結果
    plt.subplot(4, 5, 4)
    channels = list(channel_results.keys())
    scores = [channel_results[ch]['score'] for ch in channels]
    colors = ['red', 'green', 'blue']
    bars = plt.bar(channels, scores, color=colors, alpha=0.7)
    plt.title('通道評分比較', fontsize=12)
    plt.ylabel('綜合評分')
    plt.xticks(rotation=45)
    
    # 標記最佳通道
    best_idx = channels.index(best_channel_name)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)
    
    # 性能指標比較
    plt.subplot(4, 5, 5)
    methods = ['Sobel方法', '通道方法']
    coverages = [sobel_metrics['vessel_coverage'], channel_metrics['vessel_coverage']]
    vessel_counts = [sobel_metrics['num_vessels'], channel_metrics['num_vessels']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, coverages, width, label='血管覆蓋率(%)', alpha=0.8)
    plt.bar(x + width/2, vessel_counts, width, label='血管數量', alpha=0.8)
    
    plt.xlabel('檢測方法')
    plt.ylabel('數值')
    plt.title('性能指標比較', fontsize=12)
    plt.xticks(x, methods)
    plt.legend()
    
    # 第二行：Sobel方法處理過程
    sobel_binary, sobel_combined, sobel_processed, sobel_eq, sobel_threshold = sobel_results
    
    plt.subplot(4, 5, 6)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel邊緣檢測', fontsize=11)
    plt.axis('off')
    
    plt.subplot(4, 5, 7)
    plt.imshow(sobel_eq, cmap='gray')
    plt.title('直方圖均衡化', fontsize=11)
    plt.axis('off')
    
    plt.subplot(4, 5, 8)
    plt.imshow(sobel_processed, cmap='gray')
    plt.title('邊緣抑制+統計移除', fontsize=11)
    plt.axis('off')
    
    plt.subplot(4, 5, 9)
    plt.imshow(sobel_binary, cmap='gray')
    plt.title('Sobel最終結果', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(4, 5, 10)
    plt.text(0.1, 0.8, f'統計閾值: {sobel_threshold}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'血管覆蓋率: {sobel_metrics["vessel_coverage"]:.2f}%', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'檢測血管數: {sobel_metrics["num_vessels"]}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'血管像素數: {sobel_metrics["vessel_pixels"]}', fontsize=10, transform=plt.gca().transAxes)
    plt.title('Sobel方法指標', fontsize=11)
    plt.axis('off')
    
    # 第三行：通道方法處理過程
    channel_binary, channel_enhanced, channel_edges, adaptive_threshold = channel_detection_results
    
    plt.subplot(4, 5, 11)
    plt.imshow(channel_enhanced, cmap='gray')
    plt.title('CLAHE增強', fontsize=11)
    plt.axis('off')
    
    plt.subplot(4, 5, 12)
    plt.imshow(channel_edges, cmap='gray')
    plt.title('邊緣處理+均衡化', fontsize=11)
    plt.axis('off')
    
    plt.subplot(4, 5, 13)
    plt.imshow(channel_binary, cmap='gray')
    plt.title(f'通道方法結果 (k=1.5)', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(4, 5, 14)
    plt.text(0.1, 0.8, f'自適應閾值: {adaptive_threshold:.2f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'血管覆蓋率: {channel_metrics["vessel_coverage"]:.2f}%', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'檢測血管數: {channel_metrics["num_vessels"]}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'血管像素數: {channel_metrics["vessel_pixels"]}', fontsize=10, transform=plt.gca().transAxes)
    plt.title('通道方法指標', fontsize=11)
    plt.axis('off')
    
    # 第四行：結果比較
    plt.subplot(4, 5, 16)
    # 創建彩色疊加圖
    overlay = cv2.cvtColor(best_channel, cv2.COLOR_GRAY2RGB)
    overlay[sobel_binary == 255] = [255, 0, 0]  # 紅色表示Sobel檢測
    plt.imshow(overlay)
    plt.title('Sobel結果疊加', fontsize=11)
    plt.axis('off')
    
    plt.subplot(4, 5, 17)
    overlay2 = cv2.cvtColor(best_channel, cv2.COLOR_GRAY2RGB)
    overlay2[channel_binary == 255] = [0, 255, 0]  # 綠色表示通道檢測
    plt.imshow(overlay2)
    plt.title('通道結果疊加', fontsize=11)
    plt.axis('off')
    
    plt.subplot(4, 5, 18)
    # 兩種方法結果比較
    comparison = cv2.cvtColor(best_channel, cv2.COLOR_GRAY2RGB)
    comparison[sobel_binary == 255] = [255, 0, 0]  # 紅色：Sobel
    comparison[channel_binary == 255] = [0, 255, 0]  # 綠色：通道
    # 重疊區域顯示為黃色
    overlap = (sobel_binary == 255) & (channel_binary == 255)
    comparison[overlap] = [255, 255, 0]
    plt.imshow(comparison)
    plt.title('方法比較\n紅:Sobel 綠:通道 黃:重疊', fontsize=10)
    plt.axis('off')
    
    plt.subplot(4, 5, 19)
    # 通道詳細指標
    channel_names = list(channel_results.keys())
    contrasts = [channel_results[ch]['contrast'] for ch in channel_names]
    edge_strengths = [channel_results[ch]['edge_strength'] for ch in channel_names]
    
    x = np.arange(len(channel_names))
    width = 0.35
    
    plt.bar(x - width/2, contrasts, width, label='對比度', alpha=0.8)
    plt.bar(x + width/2, edge_strengths, width, label='邊緣強度', alpha=0.8)
    
    plt.xlabel('通道')
    plt.ylabel('數值')
    plt.title('通道特性分析', fontsize=11)
    plt.xticks(x, channel_names, rotation=45)
    plt.legend()
    
    plt.subplot(4, 5, 20)
    # 改進效果總結
    improvements = [
        "✓ 統一使用最佳通道",
        "✓ 調整處理順序",
        "✓ 使用k=1.5提高精確度", 
        "✓ 增加性能評估指標",
        "✓ 先增強後抑制策略"
    ]
    
    for i, improvement in enumerate(improvements):
        plt.text(0.05, 0.9 - i*0.15, improvement, fontsize=10, 
                transform=plt.gca().transAxes, color='darkgreen')
    
    plt.title('改進項目', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    改進版主程式
    """
    print("=== 改進版眼底血管檢測系統 ===")
    print("主要改進:")
    print("1. 統一使用最佳通道")
    print("2. 調整處理順序（先增強再抑制）") 
    print("3. 使用k=1.5提高檢測精確度")
    print("4. 增加性能評估指標")
    print("=" * 50)
    
    # 選擇影像檔案
    initial_dir = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Quadrant_division"
    image_path = select_image_file(initial_dir)
    
    if not image_path:
        print("未選擇檔案，程式結束")
        return
    
    try:
        # 讀取並調整影像大小
        image = np.array(Image.open(image_path))
        image = resize_image(image, max_size=640)
        print(f"影像大小: {image.shape}")
        
        # 建立眼球區域遮罩
        print("\n建立眼球區域遮罩...")
        mask = create_improved_circular_mask(image)
        
        # 分析通道並選擇最佳通道
        print("\n分析RGB通道...")
        best_channel, best_channel_name, channel_results = analyze_all_channels(image, mask)

        print(f"最佳通道: {best_channel_name}")
        print("通道分析結果:")
        for name, metrics in channel_results.items():
            print(f"通道: {name}")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value}")
        print("\n開始血管檢測...")
        # Sobel方法血管檢測
        sobel_binary, sobel_combined, sobel_processed, sobel_eq, sobel_threshold = sobel_vessel_detection(image, mask, best_channel)    
        sobel_metrics = calculate_performance_metrics(sobel_binary, mask)
        print("Sobel方法性能指標:")
        for key, value in sobel_metrics.items():
            print(f"{key}: {value}")
        # 通道方法血管檢測
        channel_binary, channel_enhanced, channel_edges, adaptive_threshold = channel_vessel_detection(image, mask, best_channel)
        channel_metrics = calculate_performance_metrics(channel_binary, mask)
        print("通道方法性能指標:")
        for key, value in channel_metrics.items():
            print(f"{key}: {value}")    
        # 顯示結果
        display_results(image, mask, best_channel, best_channel_name, channel_results, 
                       (sobel_binary, sobel_combined, sobel_processed, sobel_eq, sobel_threshold), 
                       (channel_binary, channel_enhanced, channel_edges, adaptive_threshold), 
                       sobel_metrics, channel_metrics)
    except Exception as e:
        print(f"處理影像時發生錯誤: {e}")
        return
    finally:
        print("程式結束，謝謝使用！")
        cv2.destroyAllWindows()
        plt.close('all')
        Tk().withdraw()  # 確保關閉Tkinter主視窗
        root = Tk()
        root.destroy()  # 確保關閉Tkinter主視窗
        root.mainloop()

if __name__ == "__main__":
    main()