import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, Tk
import tkinter as tk

def select_image_file(folder_path):
    """
    Allow user to select an image file from the specified folder
    """
    # Hide the main tkinter window
    root = Tk()
    root.withdraw()
    
    # Set initial directory to the specified folder
    if os.path.exists(folder_path):
        initial_dir = folder_path
    else:
        initial_dir = "/"
        print(f"Warning: Folder path '{folder_path}' does not exist. Using root directory.")
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Retinal Image",
        initialdir=initial_dir,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def load_and_preprocess_image(image_path):
    """
    Load image and convert from BGR to RGB
    """
    # Load image using OpenCV (BGR format)
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB for proper display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb

def calculate_rgb_histograms(image):
    """
    Calculate histograms for R, G, B channels
    """
    # Separate RGB channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    
    # Calculate histograms
    hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])
    
    return hist_r, hist_g, hist_b

def display_image_and_histograms(image, hist_r, hist_g, hist_b, image_name):
    """
    Display original image and RGB histograms
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Retinal Image Analysis: {image_name}', fontsize=16, fontweight='bold')
    
    # Display original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Retinal Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Display R channel histogram
    axes[0, 1].plot(hist_r, color='red', alpha=0.7, linewidth=2)
    axes[0, 1].set_title('Red Channel Histogram', fontsize=14)
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 256])
    
    # Display G channel histogram
    axes[1, 0].plot(hist_g, color='green', alpha=0.7, linewidth=2)
    axes[1, 0].set_title('Green Channel Histogram', fontsize=14)
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 256])
    
    # Display B channel histogram
    axes[1, 1].plot(hist_b, color='blue', alpha=0.7, linewidth=2)
    axes[1, 1].set_title('Blue Channel Histogram', fontsize=14)
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

def display_individual_channels(image, image_name):
    """
    Display individual RGB channels for better vessel visualization
    """
    # Separate RGB channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'RGB Channel Analysis: {image_name}', fontsize=16, fontweight='bold')
    
    # Display original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Display R channel
    axes[0, 1].imshow(r_channel, cmap='Reds')
    axes[0, 1].set_title('Red Channel (Best for Vessel Detection)', fontsize=14)
    axes[0, 1].axis('off')
    
    # Display G channel
    axes[1, 0].imshow(g_channel, cmap='Greens')
    axes[1, 0].set_title('Green Channel', fontsize=14)
    axes[1, 0].axis('off')
    
    # Display B channel
    axes[1, 1].imshow(b_channel, cmap='Blues')
    axes[1, 1].set_title('Blue Channel', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_histogram_statistics(hist_r, hist_g, hist_b):
    """
    Analyze and print histogram statistics with vessel detection recommendations
    """
    print("\n" + "="*60)
    print("HISTOGRAM ANALYSIS RESULTS")
    print("="*60)
    
    # Calculate statistics for each channel
    channels = ['Red', 'Green', 'Blue']
    histograms = [hist_r, hist_g, hist_b]
    vessel_thresholds = []
    
    for i, (channel, hist) in enumerate(zip(channels, histograms)):
        # Find peak intensity
        peak_intensity = np.argmax(hist)
        peak_frequency = np.max(hist)
        
        # Calculate mean intensity (weighted average)
        intensities = np.arange(256)
        mean_intensity = np.average(intensities, weights=hist.flatten())
        
        # Find intensity range containing 95% of pixels
        cumsum = np.cumsum(hist.flatten())
        total_pixels = cumsum[-1]
        
        # Find 2.5th and 97.5th percentiles
        percentile_2_5 = np.where(cumsum >= total_pixels * 0.025)[0][0]
        percentile_97_5 = np.where(cumsum >= total_pixels * 0.975)[0][0]
        
        # Calculate suggested vessel detection threshold
        vessel_threshold = int(peak_intensity * 0.7)  # 30% below peak
        vessel_thresholds.append(vessel_threshold)
        
        print(f"\n{channel} Channel Statistics:")
        print(f"  Peak Intensity: {peak_intensity}")
        print(f"  Peak Frequency: {int(peak_frequency[0])}")
        print(f"  Mean Intensity: {mean_intensity:.2f}")
        print(f"  95% Range: {percentile_2_5} - {percentile_97_5}")
        print(f"  Suggested Vessel Threshold: {vessel_threshold}")
    
    print("\n" + "="*60)
    print("VESSEL DETECTION RECOMMENDATIONS")
    print("="*60)
    print("Based on histogram analysis:")
    print(f"1. Red Channel Threshold: < {vessel_thresholds[0]} (Recommended for main vessels)")
    print(f"2. Green Channel Threshold: < {vessel_thresholds[1]} (Good contrast)")
    print(f"3. Blue Channel Threshold: < {vessel_thresholds[2]} (Fine vessels)")
    print("\nProcessing Pipeline Suggestions:")
    print("1. Use Red channel for primary vessel detection")
    print("2. Apply Gaussian blur before thresholding")
    print("3. Use morphological operations (opening/closing)")
    print("4. Consider adaptive thresholding for uneven illumination")
    print("5. Apply skeletonization for vessel centerlines")
    
    return vessel_thresholds

def apply_vessel_detection_demo(image, vessel_thresholds):
    """
    Demonstrate basic vessel detection using the calculated thresholds
    """
    # Extract red channel (best for vessel detection)
    red_channel = image[:, :, 0]
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
    
    # Apply threshold for vessel detection
    threshold_value = vessel_thresholds[0]  # Use red channel threshold
    _, vessel_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel)
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create overlay
    overlay = image.copy()
    overlay[vessel_mask > 0] = [0, 255, 0]  # Green overlay for detected vessels
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Vessel Detection Demo', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(red_channel, cmap='gray')
    axes[0, 1].set_title('Red Channel')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(vessel_mask, cmap='gray')
    axes[1, 0].set_title(f'Vessel Mask (Threshold: {threshold_value})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(result)
    axes[1, 1].set_title('Detected Vessels Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the retinal image analysis
    """
    print("Retinal Image Analysis Tool")
    print("="*40)
    
    # Define the folder path
    folder_path = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\test"
    
    try:
        # Select image file
        print("Please select a retinal image file...")
        image_path = select_image_file(folder_path)
        
        if not image_path:
            print("No file selected. Exiting...")
            return
        
        print(f"Selected image: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        print("Loading and preprocessing image...")
        image = load_and_preprocess_image(image_path)
        print(f"Image shape: {image.shape}")
        
        # Calculate RGB histograms
        print("Calculating RGB histograms...")
        hist_r, hist_g, hist_b = calculate_rgb_histograms(image)
        
        # Display results
        print("Displaying histogram analysis...")
        image_name = os.path.basename(image_path)
        display_image_and_histograms(image, hist_r, hist_g, hist_b, image_name)
        
        # Display individual channels
        print("Displaying individual RGB channels...")
        display_individual_channels(image, image_name)
        
        # Analyze histogram statistics
        vessel_thresholds = analyze_histogram_statistics(hist_r, hist_g, hist_b)
        
        # Demonstrate vessel detection
        print("\nApplying vessel detection demo...")
        apply_vessel_detection_demo(image, vessel_thresholds)
        
        print("\nAnalysis completed successfully!")
        print("\nNext steps for vessel detection:")
        print("1. Fine-tune threshold values based on your specific images")
        print("2. Experiment with different morphological operations")
        print("3. Consider using more advanced techniques like CLAHE for contrast enhancement")
        print("4. Implement vessel tracking algorithms for complete vessel tree extraction")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check the image path and file format.")

# Alternative function for batch processing
def batch_process_folder(folder_path):
    """
    Process all images in a folder and save statistics
    """
    print(f"Batch processing folder: {folder_path}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    results = []
    for image_path in image_files:
        try:
            image = load_and_preprocess_image(image_path)
            hist_r, hist_g, hist_b = calculate_rgb_histograms(image)
            
            # Calculate basic statistics
            stats = {
                'filename': os.path.basename(image_path),
                'red_peak': np.argmax(hist_r),
                'green_peak': np.argmax(hist_g),
                'blue_peak': np.argmax(hist_b),
                'red_mean': np.average(np.arange(256), weights=hist_r.flatten()),
                'green_mean': np.average(np.arange(256), weights=hist_g.flatten()),
                'blue_mean': np.average(np.arange(256), weights=hist_b.flatten())
            }
            results.append(stats)
            print(f"Processed: {stats['filename']}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Save results to CSV
    import csv
    csv_path = os.path.join(folder_path, 'histogram_analysis_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'red_peak', 'green_peak', 'blue_peak', 
                     'red_mean', 'green_mean', 'blue_mean']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to: {csv_path}")
    return results

if __name__ == "__main__":
    # You can choose between single image analysis or batch processing
    
    # For single image analysis:
    main()
    
    # For batch processing (uncomment the lines below):
    # folder_path = r"C:\Users\Zz423\Desktop\研究所\UCL\旺宏\Redina 資料\test"
    # batch_process_folder(folder_path)