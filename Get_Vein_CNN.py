import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
from datetime import datetime
import glob

# 配置類別（與訓練時保持一致）
class Config:
    # 模型參數
    IMAGE_SIZE = (640, 640)
    NUM_CLASSES = 2  # 背景 + 血管
    
    # 模型路徑
    MODEL_PATH = "models/best_vessel_cnn.pth"  # 或 best_vessel_cnn_simple.pth
    
    # 推理參數
    CONFIDENCE_THRESHOLD = 0.5  # 置信度閾值
    
    @classmethod
    def to_dict(cls):
        """將配置轉換為可序列化的字典"""
        return {
            'IMAGE_SIZE': cls.IMAGE_SIZE,
            'NUM_CLASSES': cls.NUM_CLASSES,
            'MODEL_PATH': cls.MODEL_PATH,
            'CONFIDENCE_THRESHOLD': cls.CONFIDENCE_THRESHOLD
        }

# CNN模型定義（與訓練時完全相同）
class VesselCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(VesselCNN, self).__init__()
        
        # 使用更少的通道數來節省記憶體
        self.features = nn.Sequential(
            # 第一層：RGB -> 16特徵
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

# 血管分割推理類別
class VesselSegmentationInference:
    def __init__(self, model_path=None, device=None):
        """
        初始化推理模型
        
        Args:
            model_path: 模型檔案路徑
            device: 計算設備 ('cuda' 或 'cpu')
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_info = {}
        
        print(f"🔧 初始化推理模型...")
        print(f"📱 使用設備: {self.device}")
        
        if torch.cuda.is_available() and self.device == 'cuda':
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        
        self.load_model()
    
    def load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔案: {self.model_path}")
        
        print(f"📂 載入模型: {self.model_path}")
        
        try:
            # 載入模型檢查點
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 初始化模型
            self.model = VesselCNN(num_classes=Config.NUM_CLASSES)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 儲存模型資訊
            self.model_info = {
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'best_val_iou': checkpoint.get('best_val_iou', 'Unknown'),
                'config': checkpoint.get('config', {}),
                'loaded_at': datetime.now().isoformat()
            }
            
            print(f"✅ 模型載入成功!")
            print(f"📊 訓練輪數: {self.model_info['epoch']}")
            print(f"🎯 最佳 IoU: {self.model_info['best_val_iou']:.4f}" if isinstance(self.model_info['best_val_iou'], float) else f"🎯 最佳 IoU: {self.model_info['best_val_iou']}")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
    
    def preprocess_image(self, image_path_or_array):
        """
        預處理輸入影像
        
        Args:
            image_path_or_array: 影像路徑或numpy陣列
            
        Returns:
            tensor: 預處理後的張量
            original_size: 原始影像尺寸
        """
        # 讀取影像
        if isinstance(image_path_or_array, str):
            # 從路徑讀取
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"❌ 找不到影像檔案: {image_path_or_array}")
            
            try:
                # 嘗試用OpenCV讀取
                image = cv2.imread(image_path_or_array)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # 使用PIL讀取
                    image = Image.open(image_path_or_array).convert('RGB')
                    image = np.array(image)
            except Exception as e:
                raise ValueError(f"❌ 無法讀取影像: {e}")
        else:
            # 直接使用numpy陣列
            image = image_path_or_array.copy()
        
        # 記錄原始尺寸
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # 調整尺寸到模型輸入大小
        image_resized = cv2.resize(image, Config.IMAGE_SIZE)
        
        # 正規化到 [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 轉換為張量 (H, W, C) -> (1, C, H, W)
        image_tensor = torch.FloatTensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device), original_size
    
    def postprocess_prediction(self, prediction, original_size, apply_threshold=True):
        """
        後處理預測結果
        
        Args:
            prediction: 模型預測輸出
            original_size: 原始影像尺寸 (width, height)
            apply_threshold: 是否應用置信度閾值
            
        Returns:
            dict: 包含各種格式的預測結果
        """
        # 獲取預測機率和類別
        with torch.no_grad():
            # 應用softmax獲得機率
            probabilities = torch.softmax(prediction, dim=1)
            vessel_prob = probabilities[0, 1].cpu().numpy()  # 血管機率
            
            # 獲取預測類別
            _, predicted_class = torch.max(prediction, 1)
            predicted_mask = predicted_class[0].cpu().numpy()
        
        # 調整回原始尺寸
        vessel_prob_resized = cv2.resize(vessel_prob, original_size)
        predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), original_size, 
                                          interpolation=cv2.INTER_NEAREST)
        
        # 應用置信度閾值（可選）
        if apply_threshold:
            thresholded_mask = (vessel_prob_resized > Config.CONFIDENCE_THRESHOLD).astype(np.uint8)
        else:
            thresholded_mask = predicted_mask_resized
        
        return {
            'vessel_probability': vessel_prob_resized,  # 血管機率圖 [0-1]
            'predicted_mask': predicted_mask_resized,   # 預測遮罩 [0,1]
            'thresholded_mask': thresholded_mask,       # 閾值化遮罩 [0,1]
            'vessel_probability_uint8': (vessel_prob_resized * 255).astype(np.uint8),  # 8位機率圖
            'binary_mask_uint8': (thresholded_mask * 255).astype(np.uint8)  # 8位二值遮罩
        }
    
    def predict_single_image(self, image_path_or_array, save_results=False, output_dir="results"):
        """
        對單張影像進行血管分割預測
        
        Args:
            image_path_or_array: 影像路徑或numpy陣列
            save_results: 是否儲存結果
            output_dir: 結果儲存目錄
            
        Returns:
            dict: 預測結果
        """
        if self.model is None:
            raise RuntimeError("❌ 模型尚未載入!")
        
        print(f"🔍 處理影像...")
        
        # 預處理
        image_tensor, original_size = self.preprocess_image(image_path_or_array)
        
        # 推理
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        # 後處理
        results = self.postprocess_prediction(prediction, original_size)
        
        # 計算統計資訊
        vessel_pixels = np.sum(results['thresholded_mask'])
        total_pixels = results['thresholded_mask'].size
        vessel_ratio = vessel_pixels / total_pixels
        
        results['statistics'] = {
            'vessel_pixels': int(vessel_pixels),
            'total_pixels': int(total_pixels),
            'vessel_ratio': float(vessel_ratio),
            'image_size': original_size
        }
        
        print(f"✅ 預測完成!")
        print(f"📊 血管像素: {vessel_pixels:,} / {total_pixels:,} ({vessel_ratio:.2%})")
        
        # 儲存結果（可選）
        if save_results:
            self.save_prediction_results(image_path_or_array, results, output_dir)
        
        return results
    
    def predict_batch_images(self, image_paths, save_results=False, output_dir="results"):
        """
        批次處理多張影像
        
        Args:
            image_paths: 影像路徑列表
            save_results: 是否儲存結果
            output_dir: 結果儲存目錄
            
        Returns:
            list: 所有預測結果
        """
        print(f"🔄 批次處理 {len(image_paths)} 張影像...")
        
        all_results = []
        for i, image_path in enumerate(image_paths):
            print(f"\\n處理 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                result = self.predict_single_image(image_path, save_results, output_dir)
                result['image_path'] = image_path
                all_results.append(result)
            except Exception as e:
                print(f"❌ 處理失敗: {e}")
                continue
        
        print(f"\\n🎉 批次處理完成! 成功處理 {len(all_results)}/{len(image_paths)} 張影像")
        return all_results
    
    def save_prediction_results(self, image_path_or_array, results, output_dir="results"):
        """儲存預測結果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 獲取檔案名稱
        if isinstance(image_path_or_array, str):
            base_name = os.path.splitext(os.path.basename(image_path_or_array))[0]
        else:
            base_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 儲存各種格式的結果
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_vessel_prob.png"), 
                   results['vessel_probability_uint8'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary_mask.png"), 
                   results['binary_mask_uint8'])
        
        # 儲存統計資訊
        stats_file = os.path.join(output_dir, f"{base_name}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'statistics': results['statistics'],
                'model_info': self.model_info,
                'prediction_time': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"💾 結果已儲存到: {output_dir}")
    
    def visualize_prediction(self, image_path_or_array, results=None, figsize=(15, 5)):
        """
        視覺化預測結果
        
        Args:
            image_path_or_array: 原始影像
            results: 預測結果（如果為None則重新預測）
            figsize: 圖片大小
        """
        # 如果沒有提供結果，則進行預測
        if results is None:
            results = self.predict_single_image(image_path_or_array)
        
        # 讀取原始影像
        if isinstance(image_path_or_array, str):
            original_image = cv2.imread(image_path_or_array)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            title_suffix = os.path.basename(image_path_or_array)
        else:
            original_image = image_path_or_array
            title_suffix = "Array Input"
        
        # 創建視覺化
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 原始影像
        axes[0].imshow(original_image)
        axes[0].set_title(f'原始影像\\n{title_suffix}')
        axes[0].axis('off')
        
        # 血管機率圖
        im1 = axes[1].imshow(results['vessel_probability'], cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'血管機率圖\\n最大值: {results["vessel_probability"].max():.3f}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 二值化遮罩
        axes[2].imshow(results['thresholded_mask'], cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'血管分割結果\\n血管比例: {results["statistics"]["vessel_ratio"]:.2%}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

# 便捷函數
def quick_predict(image_path, model_path=None, visualize=True, save_results=False):
    """
    快速預測單張影像
    
    Args:
        image_path: 影像路徑
        model_path: 模型路徑（可選）
        visualize: 是否顯示結果
        save_results: 是否儲存結果
        
    Returns:
        預測結果
    """
    # 初始化推理模型
    inferencer = VesselSegmentationInference(model_path)
    
    # 進行預測
    results = inferencer.predict_single_image(image_path, save_results=save_results)
    
    # 視覺化（可選）
    if visualize:
        inferencer.visualize_prediction(image_path, results)
    
    return results

def batch_predict(image_dir, model_path=None, save_results=True, output_dir="batch_results"):
    """
    批次預測資料夾中的所有影像
    
    Args:
        image_dir: 影像資料夾路徑
        model_path: 模型路徑（可選）
        save_results: 是否儲存結果
        output_dir: 結果儲存目錄
        
    Returns:
        所有預測結果
    """
    # 找到所有影像檔案
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', 
                       '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f"❌ 在 {image_dir} 中找不到任何影像檔案!")
        return []
    
    print(f"📁 找到 {len(image_paths)} 張影像")
    
    # 初始化推理模型
    inferencer = VesselSegmentationInference(model_path)
    
    # 批次預測
    results = inferencer.predict_batch_images(image_paths, save_results, output_dir)
    
    return results

# 使用範例
if __name__ == "__main__":
    print("🩸 血管分割推理模型")
    print("=" * 50)
    
    # 範例1: 快速預測單張影像
    print("\\n📝 使用範例:")
    print("1. 快速預測單張影像:")
    print("   results = quick_predict('path/to/your/image.jpg')")
    
    print("\\n2. 批次預測資料夾:")
    print("   results = batch_predict('path/to/image/folder')")
    
    print("\\n3. 自定義推理:")
    print("   inferencer = VesselSegmentationInference('models/best_vessel_cnn.pth')")
    print("   results = inferencer.predict_single_image('image.jpg', save_results=True)")
    print("   inferencer.visualize_prediction('image.jpg', results)")
    
    print("\\n4. 檢查可用的模型檔案:")
    model_files = glob.glob("models/*.pth")
    if model_files:
        print("   可用模型:")
        for model_file in model_files:
            print(f"   - {model_file}")
    else:
        print("   ❌ 在 models/ 目錄中找不到 .pth 檔案")
        print("   請確保訓練完成並且模型已儲存!")