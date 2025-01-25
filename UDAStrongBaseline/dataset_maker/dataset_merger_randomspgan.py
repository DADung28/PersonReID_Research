import os
import random
import shutil

# Define paths
base_path = '/home/jun/ReID_Dataset'
dataset_names = ['Market-1501-v15.09.15-spgan']
dataset_names.append('Market-1501-v15.09.15')
new_dataset_name = 'Market-1501-v15.09.15-spgan-random'
new_dataset_path = os.path.join(base_path, new_dataset_name, 'bounding_box_train')

# Create new dataset directory and the bounding_box_train subdirectory if they do not exist
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# Get list of images from the first dataset as a reference for common image names
reference_cam_path = os.path.join(base_path, 'Market-1501-v15.09.15', 'bounding_box_train')
if not os.path.exists(reference_cam_path):
    raise FileNotFoundError(f"Path does not exist: {reference_cam_path}")

# List all images in the reference cam directory
image_names = os.listdir(reference_cam_path)

# Iterate through each image name
for image_name in image_names:
    # Collect all possible paths for the current image name from each dataset
    possible_paths = [
        os.path.join(base_path, dataset_name, 'bounding_box_train', image_name)
        for dataset_name in dataset_names
    ]

    # Filter out non-existent paths
    existing_paths = [path for path in possible_paths if os.path.exists(path)]

    # Randomly select one of the existing paths
    selected_path = random.choice(existing_paths)

    # Copy the selected image to the new dataset directory
    dst_path = os.path.join(new_dataset_path, image_name)
    shutil.copy(selected_path, dst_path)

print("New dataset created at:", new_dataset_path)

# Define paths
base_path = '/home/jun/ReID_Dataset'
dataset_names = ['DukeMTMC-reID-spgan']
dataset_names.append('DukeMTMC-reID')
new_dataset_name = 'DukeMTMC-reID-spgan-random'
new_dataset_path = os.path.join(base_path, new_dataset_name, 'bounding_box_train')

# Create new dataset directory and the bounding_box_train subdirectory if they do not exist
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# Get list of images from the first dataset as a reference for common image names
reference_cam_path = os.path.join(base_path, 'DukeMTMC-reID-spgan', 'bounding_box_train')
if not os.path.exists(reference_cam_path):
    raise FileNotFoundError(f"Path does not exist: {reference_cam_path}")

# List all images in the reference cam directory
image_names = os.listdir(reference_cam_path)

# Iterate through each image name
for image_name in image_names:
    # Collect all possible paths for the current image name from each dataset
    possible_paths = [
        os.path.join(base_path, dataset_name, 'bounding_box_train', image_name)
        for dataset_name in dataset_names
    ]

    # Filter out non-existent paths
    existing_paths = [path for path in possible_paths if os.path.exists(path)]

    # Randomly select one of the existing paths
    selected_path = random.choice(existing_paths)

    # Copy the selected image to the new dataset directory
    dst_path = os.path.join(new_dataset_path, image_name)
    shutil.copy(selected_path, dst_path)

print("New dataset created at:", new_dataset_path)
