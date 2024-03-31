import cv2
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread(r'D:\new img lab\abc.jpg', cv2.IMREAD_GRAYSCALE)
# Canny edge detection
edges_canny = cv2.Canny(image, 100, 200)
# Prewitt edge detection
kernelx_prewitt = cv2.getDerivKernels(1, 0, 3)
kernely_prewitt = cv2.getDerivKernels(0, 1, 3)
edges_prewitt_x = cv2.filter2D(image, -1, kernelx_prewitt[0] * kernely_prewitt[0].T)
edges_prewitt_y = cv2.filter2D(image, -1, kernely_prewitt[0] * kernelx_prewitt[0].T)
edges_prewitt = edges_prewitt_x + edges_prewitt_y
# Sobel edge detection
edges_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
edges_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(edges_sobel_x, edges_sobel_y)
# Plotting18
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(edges_canny, cmap='gray')
axes[1].set_title('Canny Edge Detection')
axes[1].axis('off')
axes[2].imshow(edges_prewitt, cmap='gray')
axes[2].set_title('Prewitt Edge Detection')
axes[2].axis('off')
axes[3].imshow(edges_sobel, cmap='gray')
axes[3].set_title('Sobel Edge Detection')
axes[3].axis('off')
plt.tight_layout()
plt.show()