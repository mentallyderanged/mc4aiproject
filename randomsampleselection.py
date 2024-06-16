import os
import shutil
import random


# Set the source directory containing the images
source_dir = input("Enter the path to the source directory: ")
new_dataset_dir = input("Enter the desired path to the sample dataset directory: ")

os.makedirs(new_dataset_dir, exist_ok=True)

for folders in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folders)
    
    if os.path.isdir(folder_path):
        new_folder_path = os.path.join(new_dataset_dir, folders)
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Get a list of all image files in the source folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly select 1000 images
        selected_images = random.sample(image_files, 1000) if len(image_files) >= 1000 else image_files
        
        # Copy the selected images to the new dataset folder
        for image in selected_images:
            source_image_path = os.path.join(folder_path, image)
            destination_image_path = os.path.join(new_folder_path, image)
            shutil.copy(source_image_path, destination_image_path)