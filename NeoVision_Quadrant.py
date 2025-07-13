from ultralytics import YOLO
import cv2
import math
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import datetime

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
    import datetime
    
    # 設定保存路徑
    base_save_path = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Quadrant_division"
    
    # 獲取檔案名稱（不含副檔名）
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 創建以檔案名稱命名的資料夾
    folder_path = os.path.join(base_save_path, filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"創建資料夾：{folder_path}")
    else:
        print(f"使用現有資料夾：{folder_path}")
    
    # 獲取當前日期
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 保存每個象限圖片
    saved_files = []
    for q_name, q_image in quadrants.items():
        # 生成檔案名稱：象限名稱 + 日期 + .jpg
        save_filename = f"{q_name}_{current_date}.jpg"
        save_filepath = os.path.join(folder_path, save_filename)
        
        # 保存圖片
        success = cv2.imwrite(save_filepath, q_image)
        if success:
            saved_files.append(save_filepath)
            print(f"已保存：{save_filepath}")
        else:
            print(f"保存失敗：{save_filepath}")
    
    print(f"共保存了 {len(saved_files)} 個象限圖片")
    return saved_files

def process_zone_detection(image_path):
    """處理區域檢測並顯示結果"""
    # 載入區域檢測模型
    zone_model = YOLO(r"C:\\Users\\Zz423\\Desktop\\研究所\\UCL\\旺宏\\Redina 資料\\ROP-zones\\runs\\segment\\ROP6\\weights\\best.pt")
    
    # 讀取圖片
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"錯誤：無法讀取圖片 - {image_path}")
        return False
    
    # 複製原圖進行繪製
    image = original_image.copy()
    
    # 使用YOLO進行檢測
    results = zone_model(image_path)
    
    centers = []  # 儲存中心點
    od_center = None  # OD的中心點
    
    # 處理檢測結果
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                center = calculate_center(x1, y1, x2, y2)
                centers.append((center, class_id))
                if class_id == 1:  # OD class
                    od_center = center
    
    # 繪製結果
    if len(centers) == 2:
        center1, class_id1 = centers[0]
        center2, class_id2 = centers[1]
        
        # 繪製中心點
        cv2.circle(image, (int(center1[0]), int(center1[1])), 7, (0, 255, 0), -1)
        cv2.circle(image, (int(center2[0]), int(center2[1])), 7, (0, 0, 255), -1)
        
        # 添加標籤
        for center, class_id in centers:
            label = "OD" if class_id == 1 else "M"
            text_position = (int(center[0]), int(center[1] - 10))
            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # 計算並顯示距離
        distance = calculate_distance(center1, center2)
        midpoint = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        cv2.putText(image, f'Distance: {distance:.2f}', 
                   (int(midpoint[0]), int(midpoint[1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 繪製連接線
        cv2.line(image, (int(center1[0]), int(center1[1])),
                (int(center2[0]), int(center2[1])), (255, 0, 0), 2)
        
        # 在OD中心點繪製圓
        if od_center:
            circle_radius = int(distance * 2)
            cv2.circle(image, (int(od_center[0]), int(od_center[1])),
                      circle_radius, (255, 255, 0), 2)
        
        result_status = "Zones_Detected"
    else:
        result_status = "No_Zones"
    
    # 現在根據檢測得到的OD中心點對原圖進行象限切割
    quadrants = Quadrant_division(original_image.copy(), od_center)
    
    # 調試資訊
    print(f"DEBUG: 收到 {len(quadrants)} 個象限")
    for q_name in quadrants.keys():
        print(f"DEBUG: 象限 {q_name}")

    # 保存象限圖片
    if quadrants:
        print("正在保存象限圖片...")
        saved_files = save_quadrant_images(quadrants, image_path)
        print(f"象限圖片已保存至相應資料夾")
    
    # 清理可能存在的視窗
    cv2.destroyAllWindows()
    
    # 顯示主要結果
    main_window_name = f"Main_Result_{os.path.basename(image_path)}"
    cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_window_name, 640, 480)
    cv2.imshow(main_window_name, image)
    
    # 顯示四個象限，使用英文名稱避免編碼問題
    # 計算視窗位置，讓它們不重疊
    window_positions = [
        (700, 50),   # Q1 右上象限位置
        (1200, 50),  # Q2 左上象限位置  
        (700, 400),  # Q3 左下象限位置
        (1200, 400)  # Q4 右下象限位置
    ]
    
    # 中文標籤對應
    chinese_labels = {
        "Q1_TopRight": "右上",
        "Q2_TopLeft": "左上", 
        "Q3_BottomLeft": "左下",
        "Q4_BottomRight": "右下"
    }
    
    # 顯示象限
    quadrant_list = list(quadrants.items())
    print(f"DEBUG: 準備顯示 {len(quadrant_list)} 個象限視窗")
    
    for i, (q_name, q_image) in enumerate(quadrant_list):
        print(f"DEBUG: 正在顯示 {q_name}")
        print(f"DEBUG: 象限 {q_name} 顯示大小 {q_image.shape}")
        
        # 使用英文名稱創建視窗，避免編碼問題
        window_name = q_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 設定視窗初始大小為圖片的原始大小
        cv2.resizeWindow(window_name, q_image.shape[1], q_image.shape[0])
        
        # 設定視窗位置
        if i < len(window_positions):
            cv2.moveWindow(window_name, window_positions[i][0], window_positions[i][1])
        
        cv2.imshow(window_name, q_image)
        print(f"DEBUG: 視窗 {window_name} 已顯示，大小 {q_image.shape[1]}x{q_image.shape[0]}")
    
    print(f"處理完成：{os.path.basename(image_path)} - {result_status}")
    print("按任意鍵關閉視窗...")
    
    # 等待按鍵
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return len(centers) == 2

def Quadrant_division(image, od_center):
    """將影像分為四個象限 (基於原圖切割，不做任何繪製)"""
    if od_center is None:
        # 如果沒有OD中心點，使用圖片中心點
        height, width = image.shape[:2]
        od_center = (width // 2, height // 2)
    
    height, width = image.shape[:2]
    center_x, center_y = int(od_center[0]), int(od_center[1])
    
    # 確保中心點在圖片範圍內
    center_x = max(0, min(center_x, width-1))
    center_y = max(0, min(center_y, height-1))
    
    print(f"DEBUG: 圖片大小 {width}x{height}, OD中心點 ({center_x}, {center_y})")
    
    # 定義四個象限 (以OD中心點為基準，直接切割原圖) - 使用英文名稱
    quadrants = {
        "Q1_TopRight": image[0:center_y, center_x:width],        # 右上象限
        "Q2_TopLeft": image[0:center_y, 0:center_x],             # 左上象限
        "Q3_BottomLeft": image[center_y:height, 0:center_x],     # 左下象限
        "Q4_BottomRight": image[center_y:height, center_x:width] # 右下象限
    }
    
    # 不進行縮放，直接返回原始大小的象限
    original_quadrants = {}
    for q_name, q_image in quadrants.items():
        print(f"DEBUG: 處理象限 {q_name}, 原始大小 {q_image.shape}")
        
        # 檢查象限是否有實際內容 (不是空的或太小)
        if q_image.size > 0 and q_image.shape[0] > 0 and q_image.shape[1] > 0:
            # 保持原始大小，不進行縮放
            original_quadrants[q_name] = q_image
            print(f"DEBUG: 象限 {q_name} 保持原始大小 {q_image.shape}")
        else:
            print(f"DEBUG: 象限 {q_name} 無效，跳過")
    
    print(f"DEBUG: 返回 {len(original_quadrants)} 個有效象限")
    return original_quadrants

def select_image_file(initial_dir):
    """使用檔案選擇對話框選擇圖片"""
    # 創建隱藏的 tkinter 根視窗
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    
    # 設定檔案選擇對話框
    file_types = [
        ('所有圖片格式', '*.jpg;*.jpeg;*.png;*.bmp;*.tiff'),
        ('JPEG 檔案', '*.jpg;*.jpeg'),
        ('PNG 檔案', '*.png'),
        ('BMP 檔案', '*.bmp'),
        ('TIFF 檔案', '*.tiff'),
        ('所有檔案', '*.*')
    ]
    
    # 開啟檔案選擇對話框
    selected_file = filedialog.askopenfilename(
        title="選擇要處理的圖片",
        initialdir=initial_dir,
        filetypes=file_types
    )
    
    # 關閉 tkinter
    root.destroy()
    
    return selected_file

def main():
    """主程式"""
    # 設定預設資料夾路徑
    default_folder = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\test"
    
    print("=== ROP 影像區域檢測系統（手動選擇版）===")
    print(f"預設資料夾：{default_folder}")
    print()
    
    # 檢查預設資料夾是否存在
    if not os.path.exists(default_folder):
        print(f"警告：預設資料夾不存在 - {default_folder}")
        print("將使用當前目錄作為起始位置")
        default_folder = os.getcwd()
    
    try:
        selected_image = select_image_file(default_folder)
        if selected_image:
            print(f"已選擇：{os.path.basename(selected_image)}")
            print("正在處理圖片...")
                    
            # 處理選擇的圖片
            success = process_zone_detection(selected_image)
            
            if success:
                print("✓ 成功檢測到區域")
            else:
                print("✗ 未檢測到完整區域")
            
            print("程式結束")
        else:
            print("未選擇任何圖片，程式結束")
                    
            
                
    except KeyboardInterrupt:
        print("\\n程式被使用者中斷")
    except Exception as e:
        print(f"\\n發生錯誤：{str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()