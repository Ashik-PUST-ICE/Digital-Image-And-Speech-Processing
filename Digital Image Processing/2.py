import cv2 as cv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_images(folder_path):
    """Load images from a folder."""
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    images = [cv.imread(image_path) for image_path in image_files]
    return images

def semantic_segmentation(images, output_folder, num_clusters=8):
    """Perform semantic segmentation using K-means clustering."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, img in enumerate(images):
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv.kmeans(pixels, num_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_img = centers[labels.flatten()].reshape(img.shape)
        output_path = os.path.join(output_folder, f"segmented_image_{i}.jpg")
        cv.imwrite(output_path, segmented_img)

# Define folder paths
input_folder = r"D:\DIP Photo"  
output_folder = "segmented_images"

# Load images
images = load_images(input_folder)

# Check if images were loaded correctly
if len(images) == 0:
    print("No images found in the specified folder.")
    exit()

# Perform segmentation and save segmented images
semantic_segmentation(images, output_folder)

print("Segmentation completed. Segmented images are saved in:", output_folder)
