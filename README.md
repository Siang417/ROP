# **UCL ROP 檢測系統**

## 各項系統程式碼
- NeoVision.py : 先前旺宏比賽的伺服器系統程式，包含從雲端接收最新需要檢測的病人照片，然後在做完ridge檢測、Zone1區域繪製、Plus Diesease的分類檢測後，會將檢測結果照片回傳至指定的雲端資料夾作保留
- NeoVision_Quadrant.py : 新增功能的系統程式，使用Yolov8n-seg進行視盤和黃斑點切割的模型，加上可以依據視盤中心點做四個象限切割的系統

## 所需的相關權重
- best-plus.pt : Plus Diesease的分類模型權重檔
- best1-DLR.pt : 視盤和黃斑點切割模型的權重檔
- best.pt      : Ridge檢測模型的權重檔