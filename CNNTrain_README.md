# CNN血管分割系統使用說明

## 系統概述

本系統使用純CNN架構（1×1卷積核）進行視網膜血管分割，專注於像素級的色彩強度學習。系統支援多個公開資料集（CHASE_DB1、DRIVE、FIVES、HRF）的自動載入和訓練。

## 檔案結構

```
project/
├── train_vessel_cnn.py      # 訓練程式
├── inference_vessel_cnn.py  # 應用程式
├── models/                  # 模型儲存目錄
├── logs/                    # 訓練記錄目錄
└── D:/ROP_vein/            # 資料集根目錄
    ├── CHASE_DB1/
    │   ├── Images/
    │   └── Masks/
    ├── DRIVE/
    │   ├── Images/
    │   └── Masks/
    ├── FIVES/
    │   ├── Images/
    │   └── Masks/
    └── HRF/
        ├── Images/
        └── Masks/
```

## 環境需求

### Python 套件
```bash
pip install torch torchvision opencv-python numpy pillow matplotlib scikit-learn
```

### 硬體需求
- **建議**: NVIDIA GPU (CUDA支援)
- **最低**: CPU (訓練時間較長)
- **記憶體**: 至少 8GB RAM
- **儲存空間**: 至少 5GB 可用空間

## 使用方式

### 1. 訓練模型

#### 基本訓練
```bash
python train_vessel_cnn.py
```

#### 自訂參數訓練
在 `train_vessel_cnn.py` 中修改 `Config` 類別：

```python
class Config:
    # 資料路徑
    DATA_ROOT = "D:/ROP_vein"  # 修改為您的資料集路徑
    DATASETS = ["CHASE_DB1", "DRIVE", "FIVES", "HRF"]
    
    # 訓練參數
    IMAGE_SIZE = (512, 512)    # 影像尺寸
    BATCH_SIZE = 4             # 批次大小
    LEARNING_RATE = 1e-4       # 學習率
    EPOCHS = 100               # 訓練輪數
    
    # 資料分割比例
    TRAIN_RATIO = 0.8          # 訓練集比例
    VAL_RATIO = 0.1            # 驗證集比例
    TEST_RATIO = 0.1           # 測試集比例
```

#### 訓練輸出
- **模型檔案**: `models/best_vessel_cnn.pth`
- **訓練記錄**: `logs/training_log.json`
- **即時輸出**: 每個epoch的損失值和評估指標

### 2. 應用模型

#### 單張影像預測
```bash
python inference_vessel_cnn.py --model models/best_vessel_cnn.pth --input test_image.jpg --output results/ --visualize
```

#### 批次影像處理
```bash
python inference_vessel_cnn.py --model models/best_vessel_cnn.pth --input input_folder/ --output results/
```

#### 參數說明
- `--model`: 訓練好的模型路徑
- `--input`: 輸入影像路徑或目錄
- `--output`: 輸出結果目錄
- `--visualize`: 顯示視覺化結果（僅適用於單張影像）

### 3. 程式化使用

#### 訓練範例
```python
from train_vessel_cnn import train_model

# 執行訓練
model, test_results = train_model()

# 查看測試結果
print("測試集結果:")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")
```

#### 推理範例
```python
from inference_vessel_cnn import VesselSegmentationInference

# 初始化推理器
inferencer = VesselSegmentationInference("models/best_vessel_cnn.pth")

# 單張影像預測
prediction, original_image = inferencer.predict("test_image.jpg")

# 視覺化結果
inferencer.visualize_results("test_image.jpg", "result.png")

# 批次處理
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = inferencer.predict_batch(image_paths, "output_dir/")
```

## 模型架構

### CNN網路結構
```
輸入: RGB影像 (3通道)
├── Conv2d(3→32, 1×1) + BatchNorm + ReLU
├── Conv2d(32→64, 1×1) + BatchNorm + ReLU
├── Conv2d(64→128, 1×1) + BatchNorm + ReLU
├── Conv2d(128→64, 1×1) + BatchNorm + ReLU
├── Conv2d(64→32, 1×1) + BatchNorm + ReLU
└── Conv2d(32→2, 1×1)
輸出: 2類別機率圖 (背景/血管)
```

### 設計特點
1. **純1×1卷積**: 專注於像素級色彩特徵學習
2. **無空間相關性**: 避免過度依賴空間結構
3. **輕量化設計**: 參數量少，推理速度快
4. **端到端訓練**: 直接從RGB到分割結果

## 評估指標

系統提供以下評估指標：

1. **IoU (Intersection over Union)**: 交集除以聯集
2. **Dice係數**: 2×交集除以總和
3. **Sensitivity (召回率)**: 真陽性率
4. **Specificity (特異性)**: 真陰性率
5. **Accuracy (準確率)**: 總體正確率

## 輸出結果

### 訓練輸出
- `best_vessel_cnn.pth`: 最佳模型檔案
- `training_log.json`: 完整訓練記錄

### 推理輸出
- `*_vessel_mask.png`: 二值化血管遮罩
- `*_vessel_prob.png`: 血管機率圖
- `*_overlay.png`: 原始影像與預測結果疊加
- `*_visualization.png`: 完整視覺化結果

## 故障排除

### 常見問題

#### 1. CUDA記憶體不足
```python
# 減少批次大小
Config.BATCH_SIZE = 2  # 或更小

# 減少影像尺寸
Config.IMAGE_SIZE = (256, 256)
```

#### 2. 資料集路徑錯誤
```python
# 確認資料集路徑正確
Config.DATA_ROOT = "您的實際路徑"

# 檢查資料夾結構
print(os.listdir(Config.DATA_ROOT))
```

#### 3. 模型載入失敗
```python
# 檢查模型檔案是否存在
import os
print(os.path.exists("models/best_vessel_cnn.pth"))

# 檢查模型檔案完整性
checkpoint = torch.load("models/best_vessel_cnn.pth", map_location='cpu')
print(checkpoint.keys())
```

#### 4. 訓練收斂慢
```python
# 調整學習率
Config.LEARNING_RATE = 1e-3  # 增加學習率

# 調整類別權重
class_weights = torch.FloatTensor([0.05, 0.95])  # 更高的血管權重
```

### 效能優化

#### 1. 訓練加速
- 使用GPU訓練
- 增加批次大小（在記憶體允許範圍內）
- 使用混合精度訓練

#### 2. 推理加速
- 使用較小的輸入尺寸
- 批次處理多張影像
- 模型量化

## 進階使用

### 自訂資料集
```python
# 修改資料載入函數
def load_custom_dataset(data_path):
    # 實作您的資料載入邏輯
    pass

# 修改Config類別
class Config:
    DATA_ROOT = "您的資料集路徑"
    DATASETS = ["您的資料集名稱"]
```

### 模型改進
```python
# 添加更多層
class ImprovedVesselCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 添加更深的網路結構
        # 或嘗試不同的卷積核大小
```

### 資料增強
```python
# 在VesselDataset中添加資料增強
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
])
```

## 技術支援

如果遇到問題，請檢查：

1. **環境配置**: Python版本、套件版本
2. **資料格式**: 影像和遮罩的格式與配對
3. **硬體資源**: GPU記憶體、系統記憶體
4. **檔案權限**: 讀寫權限設定

## 版本資訊

- **版本**: 1.0.0
- **更新日期**: 2025年7月
- **相容性**: Python 3.7+, PyTorch 1.8+
- **測試環境**: Windows 10, Ubuntu 20.04

## 授權資訊

本程式碼供學術研究使用，請遵循相關資料集的使用條款。