import cv2
import os

def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):  # Iterating through the files in the folder
        img_path = os.path.join(folder_path, filename)  # Creating full path to the image
        if os.path.isfile(img_path):  # Checking if it's a file
            img = cv2.imread(img_path)  # Reading the image
            if img is not None:  # Checking if the image was read successfully
                images.append((filename, img))  # Storing the filename and image in a list
            else:
                print(f"Error reading image: {img_path}")  # Print error message if image reading failed
    return images  # Returning the list of images

def display_and_print_images(images):
    for filename, img in images:
        print(f"Image: {filename}")  # Printing the filename
        cv2.imshow(filename, img)  # Displaying the image
        cv2.waitKey(0)  # Waiting for a key press
        cv2.destroyAllWindows()  # Closing the window

if __name__ == "__main__":
    # Set your predefined folder path here
    folder_path = "Image Input"
    images = read_images_from_folder(folder_path)  # Reading images from the folder
    print(f"Number of images read: {len(images)}")
    display_and_print_images(images)  # Displaying and printing images
    
    
    # Resizing this image and save on folder 
    
    
