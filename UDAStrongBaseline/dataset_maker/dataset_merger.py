import os
import shutil

# Define datasets
datasets = [
    {
        'base_original_dir': '/home/jun/ReID_Dataset/Market-1501-v15.09.15',
        'base_transferred_dir': '/home/jun/ReID_Dataset/Market-1501-v15.09.15-spgan',
        'base_destination_dir': '/home/jun/ReID_Dataset/Market-1501-v15.09.15-original+transfered',
        'type': 'Market'
    },
    {
        'base_original_dir': '/home/jun/ReID_Dataset/DukeMTMC-reID',
        'base_transferred_dir': '/home/jun/ReID_Dataset/DukeMTMC-reID-spgan',
        'base_destination_dir': '/home/jun/ReID_Dataset/DukeMTMC-reID-original+transfered',
        'type': 'Duke'
    }
]
# Define subdirectories to process
subdirs = ['bounding_box_test', 'bounding_box_train', 'query']

# Function to copy files from source to destination
def copy_files(source_dir, dest_dir, dataset_type, rename=False):
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        if os.path.isfile(src_path):
            if rename:
                if dataset_type == 'Market':
                    parts = filename.split('_')
                    if len(parts) == 4:
                        new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}transfered_{parts[3]}"
                    else:
                        print(f"Skipping file {filename} due to unexpected format")
                        continue
                elif dataset_type == 'Duke':
                    parts = filename.split('_')
                    if len(parts) == 3:
                        new_filename = f"{parts[0]}_{parts[1]}_{parts[2].split('.')[0]}transfered.jpg"
                    else:
                        print(f"Skipping file {filename} due to unexpected format")
                        continue
                dest_path = os.path.join(dest_dir, new_filename)
            else:
                dest_path = os.path.join(dest_dir, filename)
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")

# Process each dataset
for dataset in datasets:
    for subdir in subdirs:
        original_dir = os.path.join(dataset['base_original_dir'], subdir)
        transferred_dir = os.path.join(dataset['base_transferred_dir'], subdir)
        destination_dir = os.path.join(dataset['base_destination_dir'], subdir)
        
        # Create the destination subdirectory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)
        
        # Copy original images
        copy_files(original_dir, destination_dir, dataset['type'])
        
        # Copy and rename transferred images
        copy_files(transferred_dir, destination_dir, dataset['type'], rename=True)

print("All files have been copied and renamed successfully.")
