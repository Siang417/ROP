import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------- U-Net模型 ----------------------------
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

# ---------------------------- 資料集 ----------------------------
class VesselDatasetTest(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, save_names=False):
        self.img_dir    = img_dir
        self.mask_dir   = mask_dir
        self.transform  = transform
        self.save_names = save_names
        self.image_names = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name     = self.image_names[idx]
        img_path = os.path.join(self.img_dir, name)
        image_pil = Image.open(img_path).convert("RGB")

        # 轉換影像為 tensor [3, H, W]
        image = self.transform(image_pil) if self.transform else transforms.ToTensor()(image_pil)

        # 載入或建立遮罩 tensor [1, H, W]
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, name)
            mask_pil  = Image.open(mask_path).convert("L")
            mask = self.transform(mask_pil) if self.transform else transforms.ToTensor()(mask_pil)
        else:
            # 假的全零遮罩
            mask = torch.zeros(1, image.shape[1], image.shape[2], dtype=torch.float32)

        if self.save_names:
            return image, mask, name
        return image, mask

import copy
# ---------------------------- 視覺化 ----------------------------
def overlay_vessel(image, mask_pred, save_path):
    """
    疊加預測血管區域於原始影像，血管區域以藍色標註
    image: torch.Tensor [3,H,W]，值域[0,1]
    mask_pred: torch.Tensor [2,H,W] 或 [1,H,W]
    save_path: 儲存路徑
    """
    # 轉成 numpy 格式
    img_np = (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8) # H,W,3
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # 取得預測遮罩
    if mask_pred.ndim == 3 and mask_pred.shape[0] == 2:
        vessel_mask = (mask_pred[1].cpu().numpy() > 0.5).astype(np.uint8)
    else:
        vessel_mask = (mask_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    # 建立白色遮罩 (BGR: [255,255,255])
    white_mask = np.zeros_like(img_bgr)
    white_mask[vessel_mask == 1] = [255, 255, 255]
    # 疊加 (原圖 70% + 白色遮罩 30%)
    overlay = cv2.addWeighted(img_bgr, 0.7, white_mask, 0.3, 0)
    cv2.imwrite(save_path, overlay)
def visualize_sample(image, mask_gt, mask_pred, save_path):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # 處理 mask_gt：可能是 1 通道（真實值）或假零遮罩
    if mask_gt is not None and mask_gt.ndim == 3 and mask_gt.shape[0] == 1:
        mask_gt_np = mask_gt[0].cpu().numpy()
    else:
        mask_gt_np = np.zeros((image_np.shape[0], image_np.shape[1]))

    # 只使用預測的血管通道
    if mask_pred.ndim == 3 and mask_pred.shape[0] == 2:
        mask_pred_np = (mask_pred[1].cpu().numpy() > 0.5).astype("float")
    else:
        mask_pred_np = (mask_pred.squeeze().cpu().numpy() > 0.5).astype("float")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title("輸入影像")
    axs[1].imshow(mask_gt_np, cmap="gray")
    axs[1].set_title("真實遮罩")
    axs[2].imshow(mask_pred_np, cmap="gray")
    axs[2].set_title("預測遮罩")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_no_gt(image, mask_pred, save_path):
    """
    image:     torch.Tensor [3,H,W]，RGB順序，值域[0,1]
    mask_pred: torch.Tensor [2,H,W] 或 [1,H,W]
    """
    # 1) 轉成 H×W×C numpy uint8
    img_np    = image.permute(1,2,0).cpu().numpy()        # RGB float32 [0,1]
    img_uint8 = (img_np * 255).astype(np.uint8)           # 仍為RGB

    # 2) 轉成OpenCV的BGR格式
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # 3) 取出二值遮罩 (0/255)
    if mask_pred.ndim == 3 and mask_pred.shape[0] == 2:
        m = (mask_pred[1].cpu().numpy() > 0.5).astype(np.uint8) * 255
    else:
        m = (mask_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 4) 遮罩轉成3通道BGR
    m_color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    # 5) 並排儲存
    comp = np.hstack([img_bgr, m_color])
    cv2.imwrite(save_path, comp)

# ---------------------------- Dice係數與準確率 ----------------------------
def dice_coeff(pred, target, epsilon=1e-6):
    # 處理形狀: [B, C, H, W] → [B, H, W]
    if pred.ndim == 4:
        pred = pred[:, 0]
    if target.ndim == 4:
        target = target[:, 0]

    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean()

def pixel_acc(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).float()
    return correct.mean()

# ---------------------------- 測試執行器 ----------------------------
def run_test(use_ground_truth=False):
    TEST_IMG_DIR  = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Quadrant_division\NoPlus1-OD"
    TEST_MASK_DIR = r"C:\Users\Zz423\Downloads\SEGMENTATION\FIVES\test\Ground Truth" if use_ground_truth else None
    CKPT_PATH     = r"D:\ROP\best_vessel_cnn.pth"
    SAVE_PRED_DIR_BASE = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Predictions2"
    SAVE_VIS_DIR_BASE  = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Visualizations2"
    SAVE_OVERLAY_DIR_BASE = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Overlay"

    # 取得今天的時間戳 (YYYYMMDD)
    from datetime import datetime
    today_str = datetime.now().strftime('%Y%m%d')
    SAVE_PRED_DIR = os.path.join(SAVE_PRED_DIR_BASE, today_str)
    SAVE_VIS_DIR = os.path.join(SAVE_VIS_DIR_BASE, today_str)
    SAVE_OVERLAY_DIR = os.path.join(SAVE_OVERLAY_DIR_BASE, today_str)

    BATCH_SIZE = 2
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    # 建立或確認時間戳資料夾
    for d in [SAVE_PRED_DIR, SAVE_VIS_DIR, SAVE_OVERLAY_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # 建立測試資料集與資料載入器
    test_ds = VesselDatasetTest(TEST_IMG_DIR, TEST_MASK_DIR, transform, save_names=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 載入模型
    model = UNet()
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    dice_total, acc_total, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for imgs, masks, names in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            if masks is not None:
                masks = masks.to(DEVICE)

            preds = model(imgs)

            for i, name in enumerate(names):
                pred_mask = (preds[i, 1] > 0.5).cpu().numpy().astype("uint8") * 255
                Image.fromarray(pred_mask).save(os.path.join(SAVE_PRED_DIR, name))

                vis_path = os.path.join(SAVE_VIS_DIR, name)
                overlay_path = os.path.join(SAVE_OVERLAY_DIR, name)
                # 疊加預測血管區域於原始影像
                overlay_vessel(imgs[i], preds[i], overlay_path)
                if TEST_MASK_DIR is None:
                    # 沒有真實遮罩 → 2欄顯示
                    visualize_no_gt(imgs[i], preds[i], vis_path)
                else:
                    # 有真實遮罩 → 3欄顯示
                    visualize_sample(imgs[i], masks[i], preds[i], vis_path)

            if masks is not None:
                dice_total += dice_coeff(preds[:, 0], masks[:, 0]).item()
                acc_total  += pixel_acc(preds[:, 0], masks[:, 0]).item()
                n_batches  += 1

    if n_batches > 0:
        print(f"\n=== 測試指標 ===")
        print(f"Dice係數 : {dice_total / n_batches:.4f}")
        print(f"像素準確率   : {acc_total  / n_batches:.4f}")
    else:
        print("已儲存預測結果，未提供真實遮罩。")

# ---------------------------- 執行 ----------------------------
if __name__ == "__main__":
    run_test()