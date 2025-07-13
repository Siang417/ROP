from ultralytics import YOLO
import sys
import os
from datetime import datetime
import shutil
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import time
import re
import cv2
import math

def calculate_center(x1, y1, x2, y2):
    """計算中心點"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def calculate_distance(point1, point2):
    """計算兩點距離"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def process_zone_detection(zone_model, image):
    """處理區域檢測，返回結果和中心點"""
    results = zone_model(image)
    
    centers = []
    od_center = None
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            center = calculate_center(x1, y1, x2, y2)
            centers.append((center, class_id))
            if class_id == 1:  # OD class
                od_center = center
    
    return centers, od_center

def process_ridge_detection(model, image):
    """處理 Ridge 檢測，返回結果和檢測框"""
    results = model.predict(
        source=image,
        conf=0.25,
        save=False  # 不儲存檔案
    )
    
    has_detection = len(results[0].boxes) > 0
    result_mark = "ROP" if has_detection else "No ROP"
    
    return results[0], result_mark

def combine_detections(image_path, ridge_results, centers, od_center):
    """合併兩種檢測結果到同一張圖片"""
    image = cv2.imread(image_path)
    
    # 繪製 Ridge 檢測結果
    for box in ridge_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        
        # 繪製矩形框（改為藍色，加粗到4）
        cv2.rectangle(image, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     (255, 0, 0), 4)  # 改為藍色(BGR: 255,0,0)，線條寬度改為4
        
        # 添加置信度標籤（改為藍色，加大字體）
        label = f'Ridge: {conf:.2f}'
        cv2.putText(image, label, 
                   (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0,  # 字體大小改為1.0
                   (255, 0, 0),  # 改為藍色
                   3)  # 文字粗細改為3
    
    # 繪製 Zone 檢測結果
    if len(centers) == 2:
        center1, class_id1 = centers[0]
        center2, class_id2 = centers[1]
        
        # 繪製中心點（綠色）
        cv2.circle(image, (int(center1[0]), int(center1[1])), 7, (0, 255, 0), -1)
        cv2.circle(image, (int(center2[0]), int(center2[1])), 7, (0, 255, 0), -1)
        
        # 添加標籤
        for center, class_id in centers:
            label = "OD" if class_id == 1 else "M"
            text_position = (int(center[0]), int(center[1] - 10))
            cv2.putText(image, label, text_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # 繪製連接線
        cv2.line(image, (int(center1[0]), int(center1[1])),
                (int(center2[0]), int(center2[1])), (255, 0, 0), 2)
        
        # 在OD中心點繪製圓
        if od_center:
            circle_radius = int(calculate_distance(center1, center2) * 2)
            cv2.circle(image, (int(od_center[0]), int(od_center[1])),
                      circle_radius, (255, 255, 0), 2)
    
    return image


def download_from_drive(service, file_id, destination_path):
    """從 Google Drive 下載檔案"""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        with open(destination_path, 'wb') as f:
            f.write(fh.read())
            f.close()
        return True
    except Exception as e:
        print(f"下載檔案時發生錯誤：{str(e)}")
        return False

def create_drive_folder(service, parent_folder_id, folder_name):
    """在 Google Drive 建立資料夾，如果已存在則直接返回ID"""
    query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    folders = results.get('files', [])
    
    if folders:
        print(f"找到現有資料夾：{folder_name}")
        return folders[0]['id'], False  # False 表示不是新建的
    else:
        print(f"建立新資料夾：{folder_name}")
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        return folder.get('id'), True  # True 表示是新建的

def upload_to_drive(service, file_path, folder_id, file_name):
    """上傳檔案到 Google Drive"""
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

def save_ridge_region(image_path, ridge_box, save_dir, number, eye_type, timestamp, scale_factor=2.0):
    """儲存並放大 ridge 區域"""
    # 讀取原始圖片
    image = cv2.imread(image_path)
    
    # 獲取 ridge 框的座標
    x1, y1, x2, y2 = map(int, ridge_box.xyxy[0])
    
    # 確保座標在圖片範圍內
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    # 裁剪 ridge 區域
    ridge_region = image[y1:y2, x1:x2]
    
    # 計算放大後的尺寸
    new_width = int(ridge_region.shape[1] * scale_factor)
    new_height = int(ridge_region.shape[0] * scale_factor)
    
    # 放大圖片
    enlarged_ridge = cv2.resize(ridge_region, (new_width, new_height), 
                              interpolation=cv2.INTER_LINEAR)
    
    # 建立檔名和儲存路徑
    ridge_filename = f"{number}_{eye_type}_ridge_{timestamp}.jpg"
    ridge_path = os.path.join(save_dir, ridge_filename)
    
    # 儲存放大後的 ridge 區域
    cv2.imwrite(ridge_path, enlarged_ridge)
    
    return ridge_path

def predict_image(ridge_model_path, plus_model_path, image_path, save_dir, service_account_file=None, drive_folder_id=None):
    try:
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # 基本檢查
        if not os.path.exists(ridge_model_path):
            print(f"錯誤：Ridge模型檔案不存在 - {ridge_model_path}")
            return
        if not os.path.exists(plus_model_path):
            print(f"錯誤：Plus模型檔案不存在 - {plus_model_path}")
            return
        if not os.path.exists(image_path):
            print(f"錯誤：圖片檔案不存在 - {image_path}")
            return
            
        # 從檔名獲取編號和眼睛類型
        image_filename = os.path.basename(image_path)
        number = get_number_from_filename(image_filename)
        eye_type = get_eye_type_from_filename(image_filename)
        
        if not number or not eye_type:
            print("錯誤：無法從檔名中提取編號或眼睛類型")
            return

        # 建立本地端的編號資料夾
        number_dir = os.path.join(save_dir, number)
        if not os.path.exists(number_dir):
            os.makedirs(number_dir)

        # 初始化 Google Drive 服務
        drive_service = None
        if service_account_file and drive_folder_id:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            drive_service = build('drive', 'v3', credentials=credentials)

        # 載入模型
        ridge_model = YOLO(ridge_model_path)
        plus_model = YOLO(plus_model_path)
        zone_model = YOLO(r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\ROP-zones\runs\segment\ROP6\weights\best.pt")

        # 執行 Ridge 檢測
        ridge_results, ridge_result = process_ridge_detection(ridge_model, image_path)
        
        # 執行 Plus Disease 檢測
        plus_results = plus_model(image_path)
        plus_classification = "no plus"
        plus_confidence = 0.0
        
        for result in plus_results:
            probs = result.probs
            if probs is not None:
                no_plus_conf = probs.data[0]
                plus_conf = probs.data[1]
                plus_classification = "no plus" if no_plus_conf > plus_conf else "plus"
                plus_confidence = max(no_plus_conf, plus_conf)
        
        # 執行 Zone 檢測
        centers, od_center = process_zone_detection(zone_model, image_path)
        
        # 合併結果
        combined_image = combine_detections(image_path, ridge_results, centers, od_center)
        
        # 建立結果檔名 (包含 Ridge 和 Plus 的結果)
        result_name = f"{number}_{eye_type}_{ridge_result}_{plus_classification}_{timestamp}.jpg"
        result_path = os.path.join(number_dir, result_name)
        cv2.imwrite(result_path, combined_image)
        
        # 如果檢測到 ridge，儲存放大的 ridge 區域
        if ridge_result == "ROP":
            for idx, box in enumerate(ridge_results.boxes):
                ridge_path = save_ridge_region(
                    image_path, 
                    box, 
                    number_dir,
                    number, 
                    eye_type, 
                    timestamp,
                    scale_factor=2.0
                )
                
                if drive_service and drive_folder_id:
                    ridge_filename = os.path.basename(ridge_path)
                    file_id = upload_to_drive(drive_service, ridge_path, drive_folder_id, ridge_filename)
                    print(f"Ridge 區域圖片已上傳，檔案ID：{file_id}")
        
        # 上傳合併後的結果
        if drive_service and drive_folder_id:
            file_id = upload_to_drive(drive_service, result_path, drive_folder_id, result_name)
            print(f"檢測結果已上傳，檔案ID：{file_id}")
        
        # 輸出檢測結果
        print(f"預測完成！")
        print(f"Ridge 檢測結果：{ridge_result}")
        print(f"Plus Disease 檢測結果：{plus_classification} (信心度：{plus_confidence:.2f})")
        print(f"Zone 檢測：{'成功' if len(centers) == 2 else '失敗'}")
        
    except Exception as e:
        print(f"發生錯誤：{str(e)}")

def get_number_from_filename(filename):
    """從檔名中提取編號"""
    match = re.match(r'(\d+)_(OS|OD)_\d{8}', filename)
    if match:
        return match.group(1)
    return None

def get_eye_type_from_filename(filename):
    """從檔名中提取眼睛類型（OS或OD）"""
    match = re.match(r'\d+_(OS|OD)_\d{8}', filename)
    if match:
        return match.group(1)
    return None

def get_latest_images(service, folder_id, processed_files):
    """獲取當天最新的 OS 和 OD 圖片"""
    today = datetime.now().strftime("%Y%m%d")
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name, createdTime)',
        orderBy='createdTime desc'
    ).execute()
    
    files = results.get('files', [])
    od_file = None
    os_file = None
    
    pattern = r'\d+_(OS|OD)_\d{8}'
    
    for file in files:
        if file['name'] in processed_files:
            continue
            
        if not re.match(pattern, file['name']):
            continue
            
        if today not in file['name']:
            continue
            
        if '_OD_' in file['name'] and not od_file:
            od_file = file
        elif '_OS_' in file['name'] and not os_file:
            os_file = file
            
        if od_file and os_file:
            break
            
    return od_file, os_file

def main():
    default_ridge_model_path = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\best2-DLR.pt"
    default_plus_model_path = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\best-plus.pt"
    default_save_dir = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\預測結果"
    temp_download_dir = os.path.join(default_save_dir, "temp_downloads")
    
    SERVICE_ACCOUNT_FILE = r"C:\Users\Zz423\Desktop\研究所\UCL\Qt Designer練習\金鑰\credentials.json"
    UPLOAD_FOLDER = '1FjGwf2-pafYkXBUXb-_9E6935siFhStW'
    DOWNLOAD_FOLDER = '1EcCXzx1e-RU7dMKrxMqIka-l3oIVW4g9'
    
    if not os.path.exists(temp_download_dir):
        os.makedirs(temp_download_dir)
    
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    drive_service = build('drive', 'v3', credentials=credentials)
    
    processed_files = set()
    folder_cache = {}  # 用於快取已創建的資料夾ID
    
    print("=== ROP 影像自動檢測系統（整合版）===")
    print("系統已啟動，開始監控雲端資料夾...")
    
    try:
        while True:
            od_file, os_file = get_latest_images(drive_service, DOWNLOAD_FOLDER, processed_files)
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}] 檢查新檔案...")
            
            if od_file or os_file:
                # 從任一檔案取得病歷號
                file_to_check = od_file if od_file else os_file
                number = get_number_from_filename(file_to_check['name'])
                
                # 檢查資料夾快取
                if number not in folder_cache:
                    # 在雲端建立或獲取資料夾
                    folder_id, is_new = create_drive_folder(drive_service, UPLOAD_FOLDER, number)
                    folder_cache[number] = folder_id
                    
                    # 在本地端建立資料夾（如果不存在）
                    local_folder = os.path.join(default_save_dir, number)
                    if not os.path.exists(local_folder):
                        os.makedirs(local_folder)
                        print(f"建立本地資料夾：{number}")
                
                current_folder_id = folder_cache[number]
            
            if od_file:
                print(f"發現新的 OD 圖片：{od_file['name']}")
                download_path = os.path.join(temp_download_dir, od_file['name'])
                if download_from_drive(drive_service, od_file['id'], download_path):
                    predict_image(default_ridge_model_path, default_plus_model_path,
                               download_path, default_save_dir,
                               service_account_file=SERVICE_ACCOUNT_FILE,
                               drive_folder_id=current_folder_id)
                    processed_files.add(od_file['name'])
                    os.remove(download_path)
            
            if os_file:
                print(f"發現新的 OS 圖片：{os_file['name']}")
                download_path = os.path.join(temp_download_dir, os_file['name'])
                if download_from_drive(drive_service, os_file['id'], download_path):
                    predict_image(default_ridge_model_path, default_plus_model_path,
                               download_path, default_save_dir,
                               service_account_file=SERVICE_ACCOUNT_FILE,
                               drive_folder_id=current_folder_id)
                    processed_files.add(os_file['name'])
                    os.remove(download_path)
            
            if not od_file and not os_file:
                print("未發現新檔案")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
    except Exception as e:
        print(f"\n發生錯誤：{str(e)}")
    finally:
        if os.path.exists(temp_download_dir):
            shutil.rmtree(temp_download_dir)

if __name__ == "__main__":
    main()
