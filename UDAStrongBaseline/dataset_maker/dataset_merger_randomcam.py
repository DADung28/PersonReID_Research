import os
import random
import shutil

#name_list = ['-singlecam','-0_0','-0_1','-0_2','-0_5','-1_0','-1_1','-1_2','-1_5','-2_0','-2_1','-2_2','-2_5','-5_0','-5_1','-5_2','-5_5']
name_list = ['-1_5','-2_0','-2_1','-2_2','-2_5','-5_0','-5_1','-5_2','-5_5']
for name in name_list:
    # Define paths
    base_path = f'/home/jun/ReID_Dataset/Market-1501-v15.09.15-stargan{name}'
    dataset_names = [f'Market-1501-v15.09.15-stargan-cam{i}' for i in range(8)]
    dataset_names.append('Market-1501-v15.09.15')
    new_dataset_name = 'Market-1501-v15.09.15-stargan-randomcam'
    new_dataset_path = os.path.join(base_path, new_dataset_name, 'bounding_box_train')

    # Create new dataset directory and the bounding_box_train subdirectory if they do not exist
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Get list of images from the first dataset as a reference for common image names
    reference_cam_path = os.path.join(base_path, 'Market-1501-v15.09.15-stargan-cam0', 'bounding_box_train')
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
    base_path = f'/home/jun/ReID_Dataset/DukeMTMC-reID-stargan{name}'
    dataset_names = [f'DukeMTMC-reID-stargan-cam{i}' for i in range(6)]
    dataset_names.append('DukeMTMC-reID')
    new_dataset_name = 'DukeMTMC-reID-stargan-randomcam'
    new_dataset_path = os.path.join(base_path, new_dataset_name, 'bounding_box_train')

    # Create new dataset directory and the bounding_box_train subdirectory if they do not exist
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Get list of images from the first dataset as a reference for common image names
    reference_cam_path = os.path.join(base_path, 'DukeMTMC-reID-stargan-cam0', 'bounding_box_train')
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
