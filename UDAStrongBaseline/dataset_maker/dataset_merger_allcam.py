import os
import shutil

name_list = ['singlecam_0_0','singlecam_0_5','singlecam_1_0','singlecam_1_5','singlecam_2_0','singlecam_2_5','0_0','0_5','1_0','1_5','2_0','2_5']
#name_list = ['-singlecam']
for name in name_list:
    # Define datasets
    datasets = [
        {
            'base_original_dir': '/home/jun/ReID_Dataset/Market-1501-v15.09.15',
            'base_transferred_dir_template': '/home/jun/ReID_Dataset/Market-1501-v15.09.15-starspgan-'+name+'/Market-1501-v15.09.15-stargan-cam{}',
            'num_transferred_dirs': 8,
            'base_destination_dir': f'/home/jun/ReID_Dataset/Market-1501-v15.09.15-starspgan-{name}/Market-1501-v15.09.15-stargan-allcam',
            'type': 'Market'
        },
        {
            'base_original_dir': '/home/jun/ReID_Dataset/DukeMTMC-reID',
            'base_transferred_dir_template': '/home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-'+name+'/DukeMTMC-reID-stargan-cam{}',
            'num_transferred_dirs': 6,
            'base_destination_dir': f'/home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-{name}/DukeMTMC-reID-stargan-allcam',
            'type': 'Duke'
        }
    ]

    # Define subdirectories to process
    subdirs = ['bounding_box_test', 'bounding_box_train', 'query']

    # Function to validate directories
    def validate_directories(directories):
        for dir_path in directories:
            if not os.path.exists(dir_path):
                print(f"Error: Directory {dir_path} does not exist.")
                return False
        return True

    # Function to log operations
    def log_operation(message, logfile='operations.log'):
        with open(logfile, 'a') as f:
            f.write(message + '\n')
        print(message)

    # Function to clean up the destination directory
    def clean_destination(dest_dir):
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)

    # Function to copy files from source to destination
    def copy_files(source_dir, dest_dir, dataset_type, rename=False, transfer_index=None):
        for filename in os.listdir(source_dir):
            src_path = os.path.join(source_dir, filename)
            if os.path.isfile(src_path):
                if rename:
                    parts = filename.split('_')
                    if dataset_type == 'Market' and len(parts) == 4:
                        new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}transfered{transfer_index}_{parts[3]}"
                    elif dataset_type == 'Duke' and len(parts) == 3:
                        new_filename = f"{parts[0]}_{parts[1]}_{parts[2].split('.')[0]}transfered{transfer_index}.jpg"
                    else:
                        log_operation(f"Skipping file {filename} due to unexpected format")
                        continue
                    dest_path = os.path.join(dest_dir, new_filename)
                else:
                    dest_path = os.path.join(dest_dir, filename)
                shutil.copy(src_path, dest_path)
                log_operation(f"Copied {src_path} to {dest_path}")

    # Process each dataset
    for dataset in datasets:
        for subdir in subdirs:
            original_dir = os.path.join(dataset['base_original_dir'], subdir)
            destination_dir = os.path.join(dataset['base_destination_dir'], subdir)

            # Validate original directory
            if not validate_directories([original_dir]):
                continue

            # Clean the destination subdirectory
            clean_destination(destination_dir)

            # Copy original images
            copy_files(original_dir, destination_dir, dataset['type'])

            for i in range(dataset['num_transferred_dirs']):
                transferred_dir = os.path.join(dataset['base_transferred_dir_template'].format(i), subdir)
                
                # Validate transferred directories
                if not validate_directories([transferred_dir]):
                    continue

                # Copy and rename transferred images
                copy_files(transferred_dir, destination_dir, dataset['type'], rename=True, transfer_index=i)

    log_operation("All files have been copied and renamed successfully.")
