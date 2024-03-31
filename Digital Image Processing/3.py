import cv2
import numpy as np

def compute_area(mask):
    # Compute the area of the region in the mask
    area = np.sum(mask / 255)  # Dividing by 255 to convert binary mask to 0s and 1s
    return area

def label_region(image, mask):
    # Convert mask to BGR format for overlaying
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Overlay the mask over the image
    overlaid_image = cv2.addWeighted(image, 0.5, mask_bgr, 0.5, 0)
    return overlaid_image

if __name__ == "__main__":
    # Load the image and the mask
    image = cv2.imread("D:\\new img lab\\abc.jpg")
    mask = cv2.imread("C:\\Users\\ashik\\OneDrive\\Desktop\\Digital Image Processing\\semantic_mask.png", cv2.IMREAD_GRAYSCALE)

    # Determine the region of the image using the mask
    region_area = compute_area(mask)
    print("Area of the region: {} pixels".format(region_area))

    # Label the region by overlapping the mask over the image
    labeled_image = label_region(image, mask)

    # Display the original image, mask, and the labeled image
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Labeled Image', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
