
import os
import shutil
import random
import cv2

# Set the source directory containing the images
source_dir = input("Enter the path to the source directory: ").strip()
new_dataset_dir = input("Enter the desired path to the sample dataset directory: ").strip()

width = int(input("Enter the desired width: "))
height = int(input("Enter the desired height: "))

# Create the new dataset directory if it doesn't exist
os.makedirs(new_dataset_dir, exist_ok=True)

for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        new_folder_path = os.path.join(new_dataset_dir, folder)
        os.makedirs(new_folder_path, exist_ok=True)

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Copy all images from the source folder
        for image in image_files:
            source_image_path = os.path.join(folder_path, image)
            destination_image_path = os.path.join(new_folder_path, image)
            try:
                img = cv2.imread(source_image_path)
                resized_img = cv2.resize(img, (width, height))
                cv2.imwrite(destination_image_path, resized_img)
                print(f"Resized and copied {image} to {new_folder_path}")
            except IOError as e:
                print(f"Could not resize and copy {image}: {e}")
