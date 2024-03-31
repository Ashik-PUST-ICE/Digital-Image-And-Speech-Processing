import cv2
import os
import numpy as np

def resize_images(input_folder, output_folder, width, height):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename))
        resized_img = cv2.resize(img, (width, height))
        cv2.imwrite(os.path.join(output_folder, filename), resized_img)

def apply_color_transform(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename))
        transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Example: Convert to grayscale
        cv2.imwrite(os.path.join(output_folder, filename), transformed_img)

def normalize_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename)).astype('float32')
        normalized_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(output_folder, filename), normalized_img * 255)

def filter_images(input_folder, output_folder, apply_high_pass=True, apply_low_pass=True):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename))

        # Apply Gaussian blur filter (Low-pass filter)
        if apply_low_pass:
            low_pass_filtered = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(os.path.join(output_folder, f"low_pass_{filename}"), low_pass_filtered)

        # Apply Laplacian filter (High-pass filter)
        if apply_high_pass:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_filtered = cv2.Laplacian(gray_img, cv2.CV_64F)
            laplacian_filtered = np.uint8(np.absolute(laplacian_filtered))
            cv2.imwrite(os.path.join(output_folder, f"high_pass_{filename}"), laplacian_filtered)

if __name__ == "__main__":
    input_folder = "C:\\Users\\ashik\\OneDrive\\Desktop\\DIP Photo"
    output_folder_resize = "resized_images"
    output_folder_color_transform = "color_transformed_images"
    output_folder_normalized = "normalized_images"
    output_folder_filtered = "filtered_images"

    # Parameters for resizing
    width = 300
    height = 200

    resize_images(input_folder, output_folder_resize, width, height)
    apply_color_transform(input_folder, output_folder_color_transform)
    normalize_images(input_folder, output_folder_normalized)
    filter_images(input_folder, output_folder_filtered, apply_high_pass=True, apply_low_pass=True)
