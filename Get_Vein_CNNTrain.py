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
    # Data path
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
    
    # CHASE_DB1 mask options
    USE_BOTH_CHASE_MASKS = True  # True: use both masks, False: use only 1stHO
    
    MODEL_SAVE_PATH = "models/"
    LOG_PATH = "logs/"
    
    @classmethod
    def to_dict(cls):
        """Convert configuration to a serializable dictionary"""
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
        """Robust image loading function"""
        try:
            # First try reading with OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
        
        try:
            # If OpenCV fails, use PIL
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
        except:
            pass
        
        raise ValueError(f"Failed to load image: {image_path}")
    
    def load_mask_robust(self, mask_path):
        """Robust mask loading function, with special handling for GIF format"""
        try:
            # Check file extension
            ext = os.path.splitext(mask_path)[1].lower()
            
            if ext == '.gif':
                # USe PIL to read GIF files
                with Image.open(mask_path) as img:
                    # Convert to grayscale
                    mask = img.convert('L')
                    return np.array(mask)
            else:
                # Use OpenCV to read other formats
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
        except Exception as e:
            print(f"Failed to load mask (OpenCV): {mask_path}, Error: {e}")
        
        try:
            # Backup option：use PIL
            with Image.open(mask_path) as img:
                mask = img.convert('L')
                return np.array(mask)
        except Exception as e:
            print(f"Failed to load mask (PIL): {mask_path}, Error: {e}")
        
        raise ValueError(f"Failed to load mask: {mask_path}")
    
    def __getitem__(self, idx):
        try:
            # Load original image
            image = self.load_image_robust(self.image_paths[idx])
            image = cv2.resize(image, self.target_size)
            
            # Load mask
            mask = self.load_mask_robust(self.mask_paths[idx])
            mask = cv2.resize(mask, self.target_size)
            
            # Binarize mask (white = vessel = 1, black = background = 0)
            mask = (mask > 127).astype(np.uint8)
            
            # Normalize image to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to PyTorch tensors
            image = torch.FloatTensor(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            mask = torch.LongTensor(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error occurred while loading data: {e}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Mask path: {self.mask_paths[idx]}")
            # Return zero tensors to avoid crash
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
    Find corresponding mask file based on dataset-specific filename rules.
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
    """Check and report file format distribution"""
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
                    # Use all found masks
                    for mask_path in matching_masks:
                        all_image_paths.append(img_path)
                        all_mask_paths.append(mask_path)
                        dataset_pairs += 1
                else:
                    # Use only the first found mask (e.g., 1stHO for CHASE_DB1)
                    all_image_paths.append(img_path)
                    all_mask_paths.append(matching_masks[0])
                    dataset_pairs += 1
            else:
                print(f"  ⚠️ No matching mask found for {os.path.basename(img_path)}")
        
        print(f"  ✅ Successfully matched: {dataset_pairs} pairs")
    
    print(f"Total matched image-mask pairs: {len(all_image_paths)}")
    
    # 檢查檔案格式
    check_file_formats(all_image_paths, all_mask_paths)
    
    return all_image_paths, all_mask_paths

# Dataset splitting function
def split_dataset(image_paths, mask_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the sum of ratios equals 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Create paired indices and shuffle
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
    
    # Confusion matrix
    cm = confusion_matrix(target, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Compute metrics
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
    """Clear GPU memory"""
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
    # Set random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Create save directories
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.LOG_PATH, exist_ok=True)
    
    # Load dataset paths
    print("Loading dataset paths...")
    image_paths, mask_paths = load_dataset_paths(Config.DATA_ROOT, Config.DATASETS)
    
    if len(image_paths) == 0:
        print("❌ No valid image-mask pairs found!")
        return None, None
    
    # Split dataset
    print("Splitting dataset...")
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        image_paths, mask_paths, Config.TRAIN_RATIO, Config.VAL_RATIO, Config.TEST_RATIO
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # Create dataloaders
    train_dataset = VesselDataset(train_images, train_masks, target_size=Config.IMAGE_SIZE)
    val_dataset = VesselDataset(val_images, val_masks, target_size=Config.IMAGE_SIZE)
    
    # Use fewer workers to save memory
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = UNet(in_channels=3, out_channels=Config.NUM_CLASSES).to(device)
    
    # Loss function (handle class imbalance)
    # Vesel pixels are fewer, give higher weight
    class_weights = torch.FloatTensor([0.1, 0.9]).to(device)  # [背景, 血管]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # Training log
    train_losses = []
    val_metrics = []
    best_val_iou = 0.0
    
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(Config.EPOCHS):
        # Clear memory
        clear_memory()
        
        # Training phase
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
        
        # Validation phase
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
        
        # Average metrics
        num_val_samples = len(val_loader.dataset)
        avg_val_iou = val_iou_total / num_val_samples
        avg_val_dice = val_dice_total / num_val_samples
        avg_val_acc = val_acc_total / num_val_samples
        
        val_metrics.append({
            'IoU': avg_val_iou,
            'Dice': avg_val_dice,
            'Accuracy': avg_val_acc
        })
        
        # Adjust learning rate
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
        
        # Save the best model - fix pickle error
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
    
    # Save training log – also fix pickle issue
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
    
    # Evaluate on test set
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
    
    # Average test metrics
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