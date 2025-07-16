import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime
import gc

# 配置類別
class Config:
    # 資料路徑
    DATA_ROOT = "D:/ROP_vein"
    DATASETS = ["CHASE_DB1", "DRIVE", "FIVES", "HRF"]
    
    # 訓練參數 - 針對RTX 4060 Laptop 8GB優化
    IMAGE_SIZE = (640, 640)
    BATCH_SIZE = 4  # 從32降低到4，避免記憶體不足
    LEARNING_RATE = 1e-4
    EPOCHS = 500
    
    # 資料分割比例
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 模型參數
    NUM_CLASSES = 2  # 背景 + 血管
    
    # CHASE_DB1 遮罩選項
    USE_BOTH_CHASE_MASKS = True  # True: 使用兩種遮罩, False: 僅使用1stHO
    
    # 儲存路徑
    MODEL_SAVE_PATH = "models/"
    LOG_PATH = "logs/"
    
    @classmethod
    def to_dict(cls):
        """將配置轉換為可序列化的字典"""
        return {
            'DATA_ROOT': cls.DATA_ROOT,
            'DATASETS': cls.DATASETS,
            'IMAGE_SIZE': cls.IMAGE_SIZE,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'LEARNING_RATE': cls.LEARNING_RATE,
            'EPOCHS': cls.EPOCHS,
            'TRAIN_RATIO': cls.TRAIN_RATIO,
            'VAL_RATIO': cls.VAL_RATIO,
            'TEST_RATIO': cls.TEST_RATIO,
            'NUM_CLASSES': cls.NUM_CLASSES,
            'USE_BOTH_CHASE_MASKS': cls.USE_BOTH_CHASE_MASKS,
            'MODEL_SAVE_PATH': cls.MODEL_SAVE_PATH,
            'LOG_PATH': cls.LOG_PATH
        }

# 自定義資料集類別
class VesselDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(640, 640)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def load_image_robust(self, image_path):
        """強化的影像讀取函數"""
        try:
            # 首先嘗試用OpenCV讀取
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
        
        try:
            # 如果OpenCV失敗，使用PIL
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
        except:
            pass
        
        raise ValueError(f"無法讀取影像: {image_path}")
    
    def load_mask_robust(self, mask_path):
        """強化的遮罩讀取函數，特別處理GIF格式"""
        try:
            # 檢查檔案副檔名
            ext = os.path.splitext(mask_path)[1].lower()
            
            if ext == '.gif':
                # 使用PIL讀取GIF檔案
                with Image.open(mask_path) as img:
                    # 轉換為灰階
                    mask = img.convert('L')
                    return np.array(mask)
            else:
                # 使用OpenCV讀取其他格式
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
        except Exception as e:
            print(f"讀取遮罩失敗 (OpenCV): {mask_path}, 錯誤: {e}")
        
        try:
            # 備用方案：使用PIL
            with Image.open(mask_path) as img:
                mask = img.convert('L')
                return np.array(mask)
        except Exception as e:
            print(f"讀取遮罩失敗 (PIL): {mask_path}, 錯誤: {e}")
        
        raise ValueError(f"無法讀取遮罩: {mask_path}")
    
    def __getitem__(self, idx):
        try:
            # 讀取原始影像
            image = self.load_image_robust(self.image_paths[idx])
            image = cv2.resize(image, self.target_size)
            
            # 讀取遮罩
            mask = self.load_mask_robust(self.mask_paths[idx])
            mask = cv2.resize(mask, self.target_size)
            
            # 二值化遮罩 (白色=血管=1, 黑色=背景=0)
            mask = (mask > 127).astype(np.uint8)
            
            # 正規化影像到 [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # 轉換為 PyTorch 張量
            image = torch.FloatTensor(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            mask = torch.LongTensor(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"讀取資料時發生錯誤: {e}")
            print(f"影像路徑: {self.image_paths[idx]}")
            print(f"遮罩路徑: {self.mask_paths[idx]}")
            # 返回零張量避免程式崩潰
            image = torch.zeros(3, self.target_size[1], self.target_size[0])
            mask = torch.zeros(self.target_size[1], self.target_size[0], dtype=torch.long)
            return image, mask

# 記憶體優化的CNN模型
class VesselCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(VesselCNN, self).__init__()
        
        # 使用更少的通道數來節省記憶體
        self.features = nn.Sequential(
            # 第一層：RGB -> 16特徵 (減少通道數)
            nn.Conv2d(3, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 第二層：16 -> 32特徵
            nn.Conv2d(16, 32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第三層：32 -> 64特徵
            nn.Conv2d(32, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第四層：64 -> 32特徵
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第五層：32 -> 16特徵
            nn.Conv2d(32, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 輸出層：16 -> 2類別 (背景/血管)
            nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        )
        
    def forward(self, x):
        return self.features(x)

# 智能檔案配對函數
def find_matching_mask(image_path, dataset_name, mask_dir):
    """
    根據不同資料集的規則尋找對應的遮罩檔案
    """
    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]
    
    matching_masks = []
    
    if dataset_name == "CHASE_DB1":
        # CHASE_DB1: Image_01L.jpg -> Image_01L_1stHO.png, Image_01L_2ndHO.png
        mask_candidates = [
            f"{image_stem}_1stHO.png",
            f"{image_stem}_2ndHO.png"
        ]
        
        for mask_name in mask_candidates:
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                matching_masks.append(mask_path)
                
    elif dataset_name == "DRIVE":
        # DRIVE: 21_training.tif -> 21_manual1.gif
        if "_training" in image_stem:
            number = image_stem.replace("_training", "")
            mask_name = f"{number}_manual1.gif"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                matching_masks.append(mask_path)
                
    elif dataset_name == "HRF":
        # HRF: 01_dr.JPG -> 01_dr.tif
        mask_name = f"{image_stem}.tif"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            matching_masks.append(mask_path)
            
    elif dataset_name == "FIVES":
        # FIVES: 假設使用相同檔名但不同副檔名
        mask_extensions = ['.png', '.tif', '.jpg', '.jpeg']
        for ext in mask_extensions:
            mask_name = f"{image_stem}{ext}"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                matching_masks.append(mask_path)
                break
    
    return matching_masks

# 檔案格式檢查函數
def check_file_formats(image_paths, mask_paths):
    """檢查並報告檔案格式分布"""
    print("\\n📊 檔案格式分析:")
    
    # 統計影像格式
    image_formats = {}
    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        image_formats[ext] = image_formats.get(ext, 0) + 1
    
    print("影像格式:")
    for ext, count in sorted(image_formats.items()):
        print(f"  {ext}: {count} 個檔案")
    
    # 統計遮罩格式
    mask_formats = {}
    for path in mask_paths:
        ext = os.path.splitext(path)[1].lower()
        mask_formats[ext] = mask_formats.get(ext, 0) + 1
    
    print("遮罩格式:")
    for ext, count in sorted(mask_formats.items()):
        print(f"  {ext}: {count} 個檔案")
    
    # 特別檢查GIF檔案
    gif_files = [path for path in mask_paths if path.lower().endswith('.gif')]
    if gif_files:
        print(f"\\n⚠️ 發現 {len(gif_files)} 個GIF遮罩檔案，將使用PIL讀取")

# 資料載入函數
def load_dataset_paths(data_root, datasets):
    all_image_paths = []
    all_mask_paths = []
    
    print("正在載入資料集...")
    
    for dataset in datasets:
        print(f"\\n處理資料集: {dataset}")
        dataset_path = os.path.join(data_root, dataset)
        image_dir = os.path.join(dataset_path, "Images")
        mask_dir = os.path.join(dataset_path, "Masks")
        
        if not os.path.exists(image_dir):
            print(f"  ❌ Images 資料夾不存在: {image_dir}")
            continue
            
        if not os.path.exists(mask_dir):
            print(f"  ❌ Masks 資料夾不存在: {mask_dir}")
            continue
        
        # 獲取所有影像檔案
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        print(f"  📸 找到 {len(image_paths)} 個影像檔案")
        
        dataset_pairs = 0
        for img_path in image_paths:
            matching_masks = find_matching_mask(img_path, dataset, mask_dir)
            
            if matching_masks:
                if dataset == "CHASE_DB1" and Config.USE_BOTH_CHASE_MASKS:
                    # 使用所有找到的遮罩
                    for mask_path in matching_masks:
                        all_image_paths.append(img_path)
                        all_mask_paths.append(mask_path)
                        dataset_pairs += 1
                else:
                    # 只使用第一個找到的遮罩（對CHASE_DB1來說是1stHO）
                    all_image_paths.append(img_path)
                    all_mask_paths.append(matching_masks[0])
                    dataset_pairs += 1
            else:
                print(f"  ⚠️ 找不到 {os.path.basename(img_path)} 對應的遮罩檔案")
        
        print(f"  ✅ 成功配對: {dataset_pairs} 組")
    
    print(f"\\n總共找到 {len(all_image_paths)} 組影像-遮罩配對")
    
    # 檢查檔案格式
    check_file_formats(all_image_paths, all_mask_paths)
    
    return all_image_paths, all_mask_paths

# 資料分割函數
def split_dataset(image_paths, mask_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 確保比例總和為1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # 創建配對的索引並打亂
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    
    n = len(indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_images = [image_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    
    val_images = [image_paths[i] for i in val_indices]
    val_masks = [mask_paths[i] for i in val_indices]
    
    test_images = [image_paths[i] for i in test_indices]
    test_masks = [mask_paths[i] for i in test_indices]
    
    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

# 評估指標計算
def calculate_metrics(pred, target):
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    # 混淆矩陣
    cm = confusion_matrix(target, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 計算各項指標
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'IoU': iou,
        'Dice': dice,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy
    }

# 記憶體清理函數
def clear_memory():
    """清理GPU記憶體"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 訓練函數
def train_model():
    # 設定隨機種子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 創建儲存目錄
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.LOG_PATH, exist_ok=True)
    
    # 載入資料集路徑
    print("載入資料集路徑...")
    image_paths, mask_paths = load_dataset_paths(Config.DATA_ROOT, Config.DATASETS)
    
    if len(image_paths) == 0:
        print("❌ 沒有找到任何有效的影像-遮罩配對！")
        return None, None
    
    # 分割資料集
    print("\\n分割資料集...")
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        image_paths, mask_paths, Config.TRAIN_RATIO, Config.VAL_RATIO, Config.TEST_RATIO
    )
    
    print(f"訓練集: {len(train_images)} 張")
    print(f"驗證集: {len(val_images)} 張")
    print(f"測試集: {len(test_images)} 張")
    
    # 創建資料載入器
    train_dataset = VesselDataset(train_images, train_masks, target_size=Config.IMAGE_SIZE)
    val_dataset = VesselDataset(val_images, val_masks, target_size=Config.IMAGE_SIZE)
    
    # 設定較少的worker數量以節省記憶體
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\n使用設備: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = VesselCNN(num_classes=Config.NUM_CLASSES).to(device)
    
    # 損失函數 (處理類別不平衡)
    # 血管像素通常較少，給予較高權重
    class_weights = torch.FloatTensor([0.1, 0.9]).to(device)  # [背景, 血管]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # 訓練記錄
    train_losses = []
    val_metrics = []
    best_val_iou = 0.0
    
    print("\\n開始訓練...")
    print("=" * 60)
    
    for epoch in range(Config.EPOCHS):
        # 清理記憶體
        clear_memory()
        
        # 訓練階段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{Config.EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 驗證階段
        model.eval()
        val_iou_total = 0.0
        val_dice_total = 0.0
        val_acc_total = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # 獲取預測結果
                _, predicted = torch.max(outputs, 1)
                
                # 計算指標
                for i in range(images.size(0)):
                    metrics = calculate_metrics(predicted[i], masks[i])
                    val_iou_total += metrics['IoU']
                    val_dice_total += metrics['Dice']
                    val_acc_total += metrics['Accuracy']
        
        # 計算平均指標
        num_val_samples = len(val_loader.dataset)
        avg_val_iou = val_iou_total / num_val_samples
        avg_val_dice = val_dice_total / num_val_samples
        avg_val_acc = val_acc_total / num_val_samples
        
        val_metrics.append({
            'IoU': avg_val_iou,
            'Dice': avg_val_dice,
            'Accuracy': avg_val_acc
        })
        
        # 學習率調整
        scheduler.step(avg_val_iou)
        
        print(f'\\nEpoch {epoch+1}/{Config.EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}, Val Acc: {avg_val_acc:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f'  GPU記憶體: 使用 {memory_used:.2f}GB, 快取 {memory_cached:.2f}GB')
        
        print('-' * 60)
        
        # 儲存最佳模型 - 修正 pickle 錯誤
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            try:
                # 使用 Config.to_dict() 而不是 Config.__dict__
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_iou': best_val_iou,
                    'config': Config.to_dict()  # 修正這裡！
                }, os.path.join(Config.MODEL_SAVE_PATH, 'best_vessel_cnn.pth'))
                print(f'✅ 新的最佳模型已儲存! IoU: {best_val_iou:.4f}')
            except Exception as e:
                print(f'⚠️ 模型儲存失敗: {e}')
                # 嘗試只儲存模型狀態
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_iou': best_val_iou
                }, os.path.join(Config.MODEL_SAVE_PATH, 'best_vessel_cnn_simple.pth'))
                print('✅ 簡化版模型已儲存!')
    
    # 儲存訓練記錄 - 也修正 pickle 問題
    training_log = {
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'best_val_iou': best_val_iou,
        'config': Config.to_dict(),  # 修正這裡！
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(os.path.join(Config.LOG_PATH, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        print("✅ 訓練記錄已儲存!")
    except Exception as e:
        print(f"⚠️ 訓練記錄儲存失敗: {e}")
    
    print(f"\\n🎉 訓練完成! 最佳驗證 IoU: {best_val_iou:.4f}")
    
    # 在測試集上評估
    print("\\n在測試集上評估...")
    test_dataset = VesselDataset(test_images, test_masks, target_size=Config.IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    model.eval()
    test_metrics = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                metrics = calculate_metrics(predicted[i], masks[i])
                test_metrics.append(metrics)
    
    # 計算平均測試指標
    avg_test_metrics = {}
    for key in test_metrics[0].keys():
        avg_test_metrics[key] = np.mean([m[key] for m in test_metrics])
    
    print("\\n📊 測試集結果:")
    print("=" * 40)
    for key, value in avg_test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 40)
    
    return model, avg_test_metrics

if __name__ == "__main__":
    model, test_results = train_model()