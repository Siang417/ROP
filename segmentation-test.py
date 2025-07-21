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
from Get_Vein_CNNTrain import UNet

# ---------------------------- Dataset ----------------------------
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

        # transform image → tensor [3, H, W]
        image = self.transform(image_pil) if self.transform else transforms.ToTensor()(image_pil)

        # load or create mask tensor [1, H, W]
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, name)
            mask_pil  = Image.open(mask_path).convert("L")
            mask = self.transform(mask_pil) if self.transform else transforms.ToTensor()(mask_pil)
        else:
            # dummy all-zeros mask
            mask = torch.zeros(1, image.shape[1], image.shape[2], dtype=torch.float32)

        if self.save_names:
            return image, mask, name
        return image, mask

# ---------------------------- Visualization ----------------------------
def visualize_sample(image, mask_gt, mask_pred, save_path):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Handle mask_gt: either 1-channel (ground truth) or dummy zero
    if mask_gt is not None and mask_gt.ndim == 3 and mask_gt.shape[0] == 1:
        mask_gt_np = mask_gt[0].cpu().numpy()
    else:
        mask_gt_np = np.zeros((image_np.shape[0], image_np.shape[1]))

    # Use only the vessel channel from prediction
    if mask_pred.ndim == 3 and mask_pred.shape[0] == 2:
        mask_pred_np = (mask_pred[1].cpu().numpy() > 0.5).astype("float")
    else:
        mask_pred_np = (mask_pred.squeeze().cpu().numpy() > 0.5).astype("float")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(mask_gt_np, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(mask_pred_np, cmap="gray")
    axs[2].set_title("Predicted Mask")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_no_gt(image, mask_pred, save_path):
    """
    image:     torch.Tensor [3,H,W] in RGB order, values [0,1]
    mask_pred: torch.Tensor [2,H,W] or [1,H,W]
    """
    # 1) to H×W×C numpy uint8
    img_np    = image.permute(1,2,0).cpu().numpy()        # RGB float32 [0,1]
    img_uint8 = (img_np * 255).astype(np.uint8)           # still RGB

    # 2) convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # 3) extract binary mask (0/255)
    if mask_pred.ndim == 3 and mask_pred.shape[0] == 2:
        m = (mask_pred[1].cpu().numpy() > 0.5).astype(np.uint8) * 255
    else:
        m = (mask_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 4) make mask 3-channel BGR
    m_color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    # 5) stack and save
    comp = np.hstack([img_bgr, m_color])
    cv2.imwrite(save_path, comp)

# ---------------------------- Dice and Accuracy ----------------------------
def dice_coeff(pred, target, epsilon=1e-6):
    # Handle shape: [B, C, H, W] → [B, H, W]
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

# ---------------------------- Test Runner ----------------------------
def run_test(use_ground_truth=False):
    TEST_IMG_DIR  = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\test"
    TEST_MASK_DIR = r"C:\Users\Zz423\Downloads\SEGMENTATION\FIVES\test\Ground Truth" if use_ground_truth else None
    CKPT_PATH     = r"D:\ROP\best_vessel_cnn.pth"
    SAVE_PRED_DIR = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Predictions2"
    SAVE_VIS_DIR  = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Visualizations2"

    BATCH_SIZE = 2
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(SAVE_PRED_DIR, exist_ok=True)
    os.makedirs(SAVE_VIS_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    test_ds = VesselDatasetTest(TEST_IMG_DIR, TEST_MASK_DIR, transform, save_names=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

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
                if TEST_MASK_DIR is None:
                    # NO ground-truth → 2-column view
                    visualize_no_gt(imgs[i], preds[i], vis_path)
                else:
                    # HAS ground-truth → 3-column view
                    visualize_sample(imgs[i], masks[i], preds[i], vis_path)

            if masks is not None:
                dice_total += dice_coeff(preds[:, 0], masks[:, 0]).item()
                acc_total  += pixel_acc(preds[:, 0], masks[:, 0]).item()
                n_batches  += 1

    if n_batches > 0:
        print(f"\n=== Test Metrics ===")
        print(f"Dice coefficient : {dice_total / n_batches:.4f}")
        print(f"Pixel accuracy   : {acc_total  / n_batches:.4f}")
    else:
        print("Predictions saved. No ground truth masks provided.")

# ---------------------------- Run ----------------------------
if __name__ == "__main__":
    run_test()