import os
import shutil
import random

# define function randomsampleselection
def randomsampleselection():
# Set the source directory containing the images
    source_dir = input("Enter the path to the source directory: ").strip()
    new_dataset_dir = input("Enter the desired path to the sample dataset directory: ").strip()
    number_of_images = int(input("Enter the number of images to be sampled per folder: ").strip())

    # Create the new dataset directory if it doesn't exist
    os.makedirs(new_dataset_dir, exist_ok=True)

    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            new_folder_path = os.path.join(new_dataset_dir, folder)
            os.makedirs(new_folder_path, exist_ok=True)

            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

            # Randomly select number of images wanted from the source folder
            selected_images = random.sample(image_files, number_of_images) if len(image_files) >= number_of_images else image_files

            # Copy the selected images to the new dataset folder
            for image in selected_images:
                source_image_path = os.path.join(folder_path, image)
                destination_image_path = os.path.join(new_folder_path, image)
                try:
                    shutil.copy(source_image_path, destination_image_path)
                    print(f"Copied {image} to {new_folder_path}")
                except IOError as e:
                    print(f"Could not copy {image}: {e}")
                    
randomsampleselection()