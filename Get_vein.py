import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import filedialog, Tk
import os
import cv2
from scipy import ndimage
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')

def select_image_file(initial_dir):
    """
    Open file dialog to select image file
    """
    root = Tk()
    root.withdraw()  # Hide main window
    
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select Retinal Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
    )
    
    root.destroy()
    return file_path

def resize_image(image, max_size=640):
    """
    Resize image to fit within max_size while maintaining aspect ratio
    """
    if len(image.shape) == 3:
        h, w = image.shape[:2]
    else:
        h, w = image.shape
    
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    return image

def create_circular_mask(image):
    """
    Create circular mask to exclude black background outside eyeball
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use threshold to separate eyeball from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (eyeball)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    return mask

def sobel_vessel_detection(image, mask):
    """
    Improved vessel detection using Sobel edge detection method
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply mask
    gray = cv2.bitwise_and(gray, mask)
    
    # Step 1: Median filtering to reduce noise
    img_median = cv2.medianBlur(gray, 3)
    
    # Step 2: Sobel edge detection
    sobel_x = cv2.Sobel(img_median, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    
    sobel_y = cv2.Sobel(img_median, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # Combine X + Y edges
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    
    # Step 3: Suppress high-intensity edges (remove strong edges like optic disc)
    sobel_processed = sobel_combined.copy()
    coords = np.column_stack(np.where(sobel_processed >= 120))
    r = 3  # radius for suppression
    
    for y, x in coords:
        x1 = max(x - r, 0)
        x2 = min(x + r + 1, sobel_processed.shape[1])
        y1 = max(y - r, 0)
        y2 = min(y + r + 1, sobel_processed.shape[0])
        sobel_processed[y1:y2, x1:x2] = 0
    
    # Step 4: Statistical pixel removal (remove lowest 50% pixel values)
    pixel_counts = np.bincount(sobel_processed.flatten(), minlength=256)
    total_pixels = sobel_processed.size
    pixels_to_remove = int(total_pixels * 0.5)
    
    cumulative_count = 0
    threshold_value = 0
    
    for pixel_value in range(256):
        cumulative_count += pixel_counts[pixel_value]
        if cumulative_count >= pixels_to_remove:
            threshold_value = pixel_value
            break
    
    sobel_processed[sobel_processed <= threshold_value] = 0
    
    # Step 5: Histogram equalization for pixels in range 5-255
    mask_eq = (sobel_processed >= 5) & (sobel_processed <= 255)
    sobel_eq = sobel_processed.copy()
    
    if np.any(mask_eq):
        roi = sobel_processed[mask_eq]
        roi_eq = cv2.equalizeHist(roi.reshape(-1, 1).astype(np.uint8)).flatten()
        sobel_eq[mask_eq] = roi_eq
    
    # Step 6: Binary thresholding (keep pixels >= 250)
    binary = np.zeros_like(sobel_eq)
    binary[sobel_eq >= 250] = 255
    
    # Step 7: Morphological operations (5 iterations of dilation + erosion)
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for better vessel preservation
    
    for i in range(3):  # Reduced iterations
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
    
    # Apply original mask to final result
    binary = cv2.bitwise_and(binary, mask)
    
    return binary, sobel_combined, sobel_processed, sobel_eq, threshold_value

def green_channel_vessel_detection(image, mask):
    """
    Traditional green channel vessel detection for comparison
    """
    if len(image.shape) == 3:
        green_channel = image[:, :, 1]  # Green channel
    else:
        green_channel = image
    
    # Apply mask
    green_channel = green_channel * (mask / 255.0)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel.astype(np.uint8))
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Adaptive threshold
    mask_bool = mask > 0
    if np.sum(mask_bool) > 0:
        mean_val = np.mean(blurred[mask_bool])
        std_val = np.std(blurred[mask_bool])
        threshold_val = max(mean_val - 1.0 * std_val, 0)
    else:
        threshold_val = 100
    
    vessel_mask = (blurred < threshold_val) & (mask > 0)
    vessel_mask = vessel_mask.astype(np.uint8) * 255
    
    # Morphological operations
    vessel_mask = morphology.binary_opening(vessel_mask, morphology.disk(1))
    vessel_mask = morphology.binary_closing(vessel_mask, morphology.disk(2))
    vessel_mask = vessel_mask.astype(np.uint8) * 255
    
    return vessel_mask, enhanced, threshold_val

def process_retinal_image():
    """
    Main processing function with improved Sobel method
    """
    print("Enhanced Retinal Vessel Segmentation System")
    print("=" * 50)
    
    # Set initial directory
    initial_dir = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\Quadrant_division"
    
    # Select image file
    print("Please select retinal image file...")
    image_path = select_image_file(initial_dir)
    
    if not image_path:
        print("No file selected, program terminated.")
        return
    
    # Load and resize image
    try:
        image = np.array(Image.open(image_path))
        print(f"Successfully loaded image: {os.path.basename(image_path)}")
        print(f"Original image size: {image.shape[1]} x {image.shape[0]} pixels")
        
        # Resize image
        image = resize_image(image, max_size=640)
        print(f"Resized image size: {image.shape[1]} x {image.shape[0]} pixels")
        
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
    
    print("\nProcessing steps:")
    print("-" * 30)
    
    # Step 1: Create circular mask
    print("Step 1: Creating circular mask...")
    mask = create_circular_mask(image)
    
    # Step 2: Sobel vessel detection
    print("Step 2: Applying improved Sobel vessel detection...")
    sobel_vessels, sobel_raw, sobel_processed, sobel_eq, threshold_val = sobel_vessel_detection(image, mask)
    
    # Step 3: Green channel detection for comparison
    print("Step 3: Applying traditional green channel detection...")
    green_vessels, green_enhanced, green_threshold = green_channel_vessel_detection(image, mask)
    
    # Step 4: Create visualizations
    print("Step 4: Creating visualizations...")
    
    # Create overlays
    sobel_overlay = np.zeros_like(image)
    sobel_overlay[sobel_vessels > 0] = [255, 0, 0]  # Red for Sobel vessels
    
    green_overlay = np.zeros_like(image)
    green_overlay[green_vessels > 0] = [0, 255, 0]  # Green for traditional vessels
    
    # Blend with original
    sobel_result = (image * 0.7 + sobel_overlay * 0.3).astype(np.uint8)
    green_result = (image * 0.7 + green_overlay * 0.3).astype(np.uint8)
    
    # Apply mask to results
    for i in range(3):
        sobel_result[:, :, i] = sobel_result[:, :, i] * (mask / 255.0)
        green_result[:, :, i] = green_result[:, :, i] * (mask / 255.0)
    
    # Display results
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Original processing steps
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Retinal Image', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Circular Mask', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sobel_raw, cmap='gray')
    axes[0, 2].set_title('Sobel Edge Detection\n(Raw)', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(sobel_processed, cmap='gray')
    axes[0, 3].set_title(f'After Edge Suppression\n& Statistical Removal', fontsize=11)
    axes[0, 3].axis('off')
    
    # Row 2: Sobel method results
    axes[1, 0].imshow(sobel_eq, cmap='gray')
    axes[1, 0].set_title('Histogram Equalized', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sobel_vessels, cmap='gray')
    axes[1, 1].set_title(f'Sobel Vessel Mask\n(Threshold: {threshold_val})', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sobel_overlay)
    axes[1, 2].set_title('Sobel Vessels Overlay\n(Red)', fontsize=11)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(sobel_result)
    axes[1, 3].set_title('Sobel Method Result', fontsize=11)
    axes[1, 3].axis('off')
    
    # Row 3: Green channel method for comparison
    axes[2, 0].imshow(green_enhanced, cmap='gray')
    axes[2, 0].set_title('Green Channel Enhanced', fontsize=11)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(green_vessels, cmap='gray')
    axes[2, 1].set_title(f'Green Channel Vessels\n(Threshold: {green_threshold:.1f})', fontsize=11)
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(green_overlay)
    axes[2, 2].set_title('Green Channel Overlay\n(Green)', fontsize=11)
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(green_result)
    axes[2, 3].set_title('Green Channel Result', fontsize=11)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Enhanced Retinal Vessel Segmentation: Sobel vs Green Channel Methods', 
                 fontsize=16, y=0.98)
    
    # Print analysis
    print("\nProcessing Summary:")
    print("-" * 30)
    print(f"• Image size: {image.shape[1]} x {image.shape[0]} pixels")
    print(f"• Sobel statistical threshold: {threshold_val}")
    print(f"• Sobel detected vessel pixels: {np.sum(sobel_vessels > 0):,}")
    print(f"• Green channel threshold: {green_threshold:.2f}")
    print(f"• Green channel detected vessel pixels: {np.sum(green_vessels > 0):,}")
    print(f"• Eyeball area: {np.sum(mask > 0):,} pixels")
    print(f"• Sobel vessel coverage: {(np.sum(sobel_vessels > 0) / np.sum(mask > 0)) * 100:.2f}%")
    print(f"• Green vessel coverage: {(np.sum(green_vessels > 0) / np.sum(mask > 0)) * 100:.2f}%")
    
    plt.show()
    
    return {
        'original_image': image,
        'mask': mask,
        'sobel_vessels': sobel_vessels,
        'green_vessels': green_vessels,
        'sobel_result': sobel_result,
        'green_result': green_result
    }

# Main execution
if __name__ == "__main__":
    results = process_retinal_image()