# CNN血管分割系統 - 快速開始指南

## 🚀 5分鐘快速開始

### 步驟1: 環境準備
```bash
# 安裝必要套件
pip install torch torchvision opencv-python numpy pillow matplotlib scikit-learn

# 創建專案目錄
mkdir vessel_segmentation
cd vessel_segmentation
```

### 步驟2: 下載程式碼
將以下兩個檔案放入專案目錄：
- `train_vessel_cnn.py` - 訓練程式
- `inference_vessel_cnn.py` - 應用程式

### 步驟3: 準備資料集
```
D:/ROP_vein/
├── CHASE_DB1/
│   ├── Images/          # 原始影像
│   └── Masks/           # 血管遮罩
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

### 步驟4: 開始訓練
```bash
python train_vessel_cnn.py
```

### 步驟5: 應用模型
```bash
# 單張影像預測
python inference_vessel_cnn.py --model models/best_vessel_cnn.pth --input test.jpg --output results/ --visualize

# 批次處理
python inference_vessel_cnn.py --model models/best_vessel_cnn.pth --input input_folder/ --output results/
```

## 📋 檢查清單

### 訓練前檢查
- [ ] Python 3.7+ 已安裝
- [ ] PyTorch 已安裝
- [ ] 資料集已準備並放置正確位置
- [ ] 有足夠的儲存空間 (至少5GB)
- [ ] 有足夠的記憶體 (至少8GB)

### 常見設定調整
```python
# 如果GPU記憶體不足，修改批次大小
Config.BATCH_SIZE = 2

# 如果訓練時間太長，減少輪數
Config.EPOCHS = 50

# 如果資料集路徑不同，修改路徑
Config.DATA_ROOT = "您的資料集路徑"
```

## 🎯 預期結果

### 訓練結果
- 訓練時間: 約2-4小時 (GPU) / 8-12小時 (CPU)
- 最終IoU: > 0.75
- 模型大小: < 10MB

### 推理結果
- 處理速度: 約1-2秒/張 (GPU) / 5-10秒/張 (CPU)
- 輸出檔案: 血管遮罩、機率圖、疊加結果

## 🔧 快速故障排除

### 問題1: CUDA記憶體不足
**解決方案**: 減少批次大小
```python
Config.BATCH_SIZE = 1  # 最小批次
```

### 問題2: 找不到資料集
**解決方案**: 檢查路徑設定
```python
import os
print(os.path.exists("D:/ROP_vein"))  # 應該返回True
```

### 問題3: 訓練速度太慢
**解決方案**: 使用GPU或減少影像尺寸
```python
Config.IMAGE_SIZE = (256, 256)  # 減小影像尺寸
```

## 📞 需要幫助？

1. 查看完整的 `README.md` 文件
2. 檢查錯誤訊息和日誌檔案
3. 確認所有依賴套件已正確安裝
4. 驗證資料集格式和路徑設定

## 🎉 成功指標

當您看到以下輸出時，表示系統運行正常：

**訓練成功**:
```
Epoch 100/100 - Train Loss: 0.1234 - Val IoU: 0.7856
✅ 訓練完成! 最佳驗證 IoU: 0.7856
模型已儲存: models/best_vessel_cnn.pth
```

**推理成功**:
```
載入模型: models/best_vessel_cnn.pth
模型載入成功! 最佳驗證 IoU: 0.7856
處理影像 1/1: test.jpg
✅ 預測完成!
```