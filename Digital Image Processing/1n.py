import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_images_from_folder(folder_path):
    images = []
    image_files = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
            image_files.append(filename)
    return images, image_files

def resize_images(images, new_size=(300, 300)):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, new_size)
        resized_images.append(resized_img)
    return resized_images

def apply_color_transform(images):
    transformed_images = []
    for img in images:
        transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        transformed_images.append(transformed_img)
    return transformed_images

def normalize_images(images):
    normalized_images = []
    for img in images:
        # Example normalization: Convert to float32 and scale to [0, 1]
        normalized_img = img.astype('float32') / 255.0
        normalized_images.append((normalized_img * 255).astype('uint8'))
    return normalized_images

folder_path = r"C:\Users\ashik\OneDrive\Desktop\DIP Photo"
output_folder = r"C:\Users\ashik\OneDrive\Desktop\DIP OutPut"

# Read images from the folder
images, image_files = read_images_from_folder(folder_path)

# Resize images
resized_images = resize_images(images)

# Apply color transform
transformed_images = apply_color_transform(images)

# Normalize images
normalized_images = normalize_images(images)

# Plot original, resized, transformed, and normalized images
fig, axes = plt.subplots(nrows=len(images), ncols=4, figsize=(20, 4*len(images)))
for i in range(len(images)):
    axes[i, 0].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(cv2.cvtColor(resized_images[i], cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title('Resized Image')
    axes[i, 1].axis('off')
    axes[i, 2].imshow(transformed_images[i], cmap='gray')
    axes[i, 2].set_title('Transformed Image')
    axes[i, 2].axis('off')
    axes[i, 3].imshow(normalized_images[i], cmap='gray')
    axes[i, 3].set_title('Normalized Image')
    axes[i, 3].axis('off')
plt.tight_layout()
plt.show()
