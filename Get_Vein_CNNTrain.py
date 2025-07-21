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
    DATA_ROOT = r"C:\Users\Redina\Downloads\SEGMENTATION"
    DATASETS = ["CHASE_DB1", "DRIVE", "FIVES", "HRF", "HVDROPDB-NEO", "HVDROPDB-RETCAM", "ROP"]
    
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 300
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    NUM_CLASSES = 2
    
    # CHASE_DB1 遮罩選項
    USE_BOTH_CHASE_MASKS = True  # True: 使用兩個遮罩, False: 只用 1stHO
    
    MODEL_SAVE_PATH = "models/"
    LOG_PATH = "logs/"
    
    @classmethod
    def to_dict(cls):
        """將配置轉換為可序列化字典"""
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

# Custom dataset class
class VesselDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def load_image_robust(self, image_path):
        """健壯的影像載入函數"""
        try:
            # 首先嘗試用 OpenCV 讀取
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
        
        try:
            # 如果 OpenCV 失敗則用 PIL
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
        except:
            pass
        
        raise ValueError(f"無法載入影像: {image_path}")
    
    def load_mask_robust(self, mask_path):
        """健壯的遮罩載入函數，特別處理 GIF 格式"""
        try:
            # 檢查檔案副檔名
            ext = os.path.splitext(mask_path)[1].lower()
            
            if ext == '.gif':
                # 用 PIL 讀取 GIF 檔
                with Image.open(mask_path) as img:
                    # 轉成灰階
                    mask = img.convert('L')
                    return np.array(mask)
            else:
                # 用 OpenCV 讀取其他格式
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
        except Exception as e:
            print(f"無法用 OpenCV 載入遮罩: {mask_path}, 錯誤: {e}")
        
        try:
            # 備用方案：用 PIL
            with Image.open(mask_path) as img:
                mask = img.convert('L')
                return np.array(mask)
        except Exception as e:
            print(f"無法用 PIL 載入遮罩: {mask_path}, 錯誤: {e}")
        
        raise ValueError(f"無法載入遮罩: {mask_path}")
    
    def __getitem__(self, idx):
        try:
            # 載入原始影像
            image = self.load_image_robust(self.image_paths[idx])
            image = cv2.resize(image, self.target_size)
            
            # 載入遮罩
            mask = self.load_mask_robust(self.mask_paths[idx])
            mask = cv2.resize(mask, self.target_size)
            
            # 遮罩二值化 (白=血管=1, 黑=背景=0)
            mask = (mask > 127).astype(np.uint8)
            
            # 影像正規化到 [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # 轉成 PyTorch tensor
            image = torch.FloatTensor(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            mask = torch.LongTensor(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error occurred while loading data: {e}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Mask path: {self.mask_paths[idx]}")
            # 回傳零 tensor 以避免程式崩潰
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
    根據資料集的檔名規則尋找對應的遮罩檔案。
    """
    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]
    matching_masks = []

    if dataset_name == "CHASE_DB1":
        mask_candidates = [
            f"{image_stem}_1stHO.png",
            f"{image_stem}_2ndHO.png"
        ]
        for mask_name in mask_candidates:
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                matching_masks.append(mask_path)

    elif dataset_name == "DRIVE":
        if "_training" in image_stem:
            number = image_stem.replace("_training", "")
            mask_name = f"{number}_manual1.gif"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                matching_masks.append(mask_path)

    elif dataset_name == "HRF":
        mask_name = f"{image_stem}.tif"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            matching_masks.append(mask_path)

    elif dataset_name == "FIVES":
        mask_extensions = ['.png', '.tif', '.jpg', '.jpeg']
        for ext in mask_extensions:
            mask_name = f"{image_stem}{ext}"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                matching_masks.append(mask_path)
                break  # Only take the first match

    elif dataset_name in ["HVDROPDB-NEO", "HVDROPDB-RETCAM"]:
        mask_name = f"{image_stem}.png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            matching_masks.append(mask_path)

    elif dataset_name == "ROP":
        mask_name = f"{image_stem}_mask.jpg"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            matching_masks.append(mask_path)

    return matching_masks

# File format checker
def check_file_formats(image_paths, mask_paths):
    """檢查並回報檔案格式分布"""
    print("📊 File Format Analysis:")
    
    # Count image formats
    image_formats = {}
    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        image_formats[ext] = image_formats.get(ext, 0) + 1
    
    print("Image formats:")
    for ext, count in sorted(image_formats.items()):
        print(f"  {ext}: {count} files")
    
    # Count mask formats
    mask_formats = {}
    for path in mask_paths:
        ext = os.path.splitext(path)[1].lower()
        mask_formats[ext] = mask_formats.get(ext, 0) + 1
    
    print("Mask formats:")
    for ext, count in sorted(mask_formats.items()):
        print(f"  {ext}: {count} files")
    
    # Special check for GIF files
    gif_files = [path for path in mask_paths if path.lower().endswith('.gif')]
    if gif_files:
        print(f"⚠️ Found {len(gif_files)} GIF mask files, will use PIL for reading")

# Dataset loading function
def load_dataset_paths(data_root, datasets):
    all_image_paths = []
    all_mask_paths = []
    
    print("Loading dataset...")
    
    for dataset in datasets:
        print(f"Processing datasets: {dataset}")
        dataset_path = os.path.join(data_root, dataset)
        image_dir = os.path.join(dataset_path, "Images")
        mask_dir = os.path.join(dataset_path, "Masks")
        
        if not os.path.exists(image_dir):
            print(f"  ❌ Image folder does not exist: {image_dir}")
            continue
            
        if not os.path.exists(mask_dir):
            print(f"  ❌ Mask folder does not exist: {mask_dir}")
            continue
        
        # 獲取所有影像檔案
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        print(f"  📸 Found {len(image_paths)} image files")
        
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
                    # 只用第一個找到的遮罩 (例如 CHASE_DB1 的 1stHO)
                    all_image_paths.append(img_path)
                    all_mask_paths.append(matching_masks[0])
                    dataset_pairs += 1
            else:
                print(f"  ⚠️ 找不到對應遮罩: {os.path.basename(img_path)}")
        
        print(f"  ✅ 成功配對: {dataset_pairs} 組")
    
    print(f"總共配對影像-遮罩組數: {len(all_image_paths)}")
    
    # 檢查檔案格式
    check_file_formats(all_image_paths, all_mask_paths)
    
    return all_image_paths, all_mask_paths

# Dataset splitting function
def split_dataset(image_paths, mask_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 確保比例總和為 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # 建立配對索引並打亂
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

# Evaluation metrics
def calculate_metrics(pred, target):
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    # 混淆矩陣
    cm = confusion_matrix(target, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 計算指標
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

# Memory cleaning function
def clear_memory():
    """清理 GPU 記憶體"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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
        bn = self.bottleneck(self.pool4(d4))

        u4 = self.up4(bn)
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

# Training function
def train_model():
    # 設定隨機種子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 建立儲存資料夾
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.LOG_PATH, exist_ok=True)
    
    # 載入資料集路徑
    print("Loading dataset paths...")
    image_paths, mask_paths = load_dataset_paths(Config.DATA_ROOT, Config.DATASETS)
    
    if len(image_paths) == 0:
        print("❌ No valid image-mask pairs found!")
        return None, None
    
    # 切分資料集
    print("Splitting dataset...")
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        image_paths, mask_paths, Config.TRAIN_RATIO, Config.VAL_RATIO, Config.TEST_RATIO
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # 建立資料載入器
    train_dataset = VesselDataset(train_images, train_masks, target_size=Config.IMAGE_SIZE)
    val_dataset = VesselDataset(val_images, val_masks, target_size=Config.IMAGE_SIZE)
    
    # 使用較少 workers 以節省記憶體
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = UNet(in_channels=3, out_channels=Config.NUM_CLASSES).to(device)
    
    # 損失函數 (處理類別不平衡)
    # 血管像素較少，給予較高權重
    class_weights = torch.FloatTensor([0.1, 0.9]).to(device)  # [背景, 血管]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # 訓練紀錄
    train_losses = []
    val_metrics = []
    best_val_iou = 0.0
    
    print("開始訓練...")
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
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Compute metrics
                for i in range(images.size(0)):
                    metrics = calculate_metrics(predicted[i], masks[i])
                    val_iou_total += metrics['IoU']
                    val_dice_total += metrics['Dice']
                    val_acc_total += metrics['Accuracy']
        
        # 平均指標
        num_val_samples = len(val_loader.dataset)
        avg_val_iou = val_iou_total / num_val_samples
        avg_val_dice = val_dice_total / num_val_samples
        avg_val_acc = val_acc_total / num_val_samples
        
        val_metrics.append({
            'IoU': avg_val_iou,
            'Dice': avg_val_dice,
            'Accuracy': avg_val_acc
        })
        
        # 調整學習率
        scheduler.step(avg_val_iou)
        
        print(f'Epoch {epoch+1}/{Config.EPOCHS}:')
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
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_iou': best_val_iou,
                    'config': Config.to_dict()
                }, os.path.join(Config.MODEL_SAVE_PATH, 'best_vessel_cnn.pth'))
                print(f'✅ New best model saved! IoU: {best_val_iou:.4f}')
            except Exception as e:
                print(f'⚠️ Failed to save model: {e}')
                # 嘗試只儲存模型狀態
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_iou': best_val_iou
                }, os.path.join(Config.MODEL_SAVE_PATH, 'best_vessel_cnn_simple.pth'))
                print('✅ Simplified model saved!')
    
    # 儲存訓練紀錄 – 也修正 pickle 問題
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
        print("✅ Training log saved!")
    except Exception as e:
        print(f"⚠️ Failed to save training log: {e}")
    
    print(f"🎉 Training completed! Best validation IoU: {best_val_iou:.4f}")
    
    # 在測試集上評估
    print("Evaluating on test set...")
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
    
    # 平均測試指標
    avg_test_metrics = {}
    for key in test_metrics[0].keys():
        avg_test_metrics[key] = np.mean([m[key] for m in test_metrics])
    
    print("📊 Test set results:")
    print("=" * 40)
    for key, value in avg_test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 40)
    
    return model, avg_test_metrics

if __name__ == "__main__":
    model, test_results = train_model()