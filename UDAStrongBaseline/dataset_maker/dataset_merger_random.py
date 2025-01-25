import os
import shutil
import random
# Define datasets
datasets = [
    {
        'base_original_dir': '/home/jun/ReID_Dataset/Market-1501-v15.09.15',
        'base_transferred_dir': '/home/jun/ReID_Dataset/Market-1501-v15.09.15-spgan',
        'base_destination_dir': '/home/jun/ReID_Dataset/Market-1501-v15.09.15-transfered_random',
        'type': 'Market'
    },
    {
        'base_original_dir': '/home/jun/ReID_Dataset/DukeMTMC-reID',
        'base_transferred_dir': '/home/jun/ReID_Dataset/DukeMTMC-reID-spgan',
        'base_destination_dir': '/home/jun/ReID_Dataset/DukeMTMC-reID-transfered_random',
        'type': 'Duke'
    }
]
# Define subdirectories to process
subdirs = ['bounding_box_test', 'bounding_box_train', 'query']
dataset_type = 'Market'
market_person_ids = []
duke_person_ids = []

for dataset in datasets:
    
   
    for subdir in subdirs:
        original_dir = os.path.join(dataset['base_original_dir'], subdir)
        transferred_dir = os.path.join(dataset['base_transferred_dir'], subdir)
        destination_dir = os.path.join(dataset['base_destination_dir'], subdir)

        for filename in os.listdir(original_dir):
            original_path = os.path.join(original_dir, filename)
            transferred_path = os.path.join(transferred_dir, filename)
            if os.path.isfile(original_path):
                if dataset['type'] == 'Market':
                    parts = filename.split('_')
                    if len(parts) == 4:
                        market_person_ids.append(parts[0])
                    else:
                        print(f"Skipping file {filename} due to unexpected format")
                        continue
                elif dataset['type'] == 'Duke':
                    parts = filename.split('_')
                    if len(parts) == 3:
                        print
                        duke_person_ids.append(parts[0])
                    else:
                        print(f"Skipping file {filename} due to unexpected format")
                        continue

market_person_ids = set(market_person_ids)
market_person_ids = sorted(market_person_ids)
duke_person_ids = set(duke_person_ids)
duke_person_ids = sorted(duke_person_ids)
probability = 0.5

for dataset in datasets:
    picked_ids = []
    non_picked_ids = []
    if dataset['type'] == 'Market': 
        for person_id in market_person_ids:
            if random.random() < probability:
                picked_ids.append(person_id)
            else:
                non_picked_ids.append(person_id)
    elif dataset['type'] == 'Duke': 
        for person_id in market_person_ids:
            if random.random() < probability:
                picked_ids.append(person_id)
            else:
                non_picked_ids.append(person_id)
    
    for subdir in subdirs:
        original_dir = os.path.join(dataset['base_original_dir'], subdir)
        transferred_dir = os.path.join(dataset['base_transferred_dir'], subdir)
        destination_dir = os.path.join(dataset['base_destination_dir'], subdir)
        os.makedirs(destination_dir, exist_ok=True)
        for filename in os.listdir(original_dir):
            original_path = os.path.join(original_dir, filename)
            transferred_path = os.path.join(transferred_dir, filename)
            dest_path = os.path.join(destination_dir, filename)
            if os.path.isfile(original_path):
                if dataset['type'] == 'Market':
                    parts = filename.split('_')
                    if len(parts) == 4:
                        if parts[0] in picked_ids:
                            shutil.copy(original_path, dest_path) 
                        else:
                            shutil.copy(transferred_path, dest_path) 
                    else:
                        print(f"Skipping file {filename} due to unexpected format")
                        continue
                elif dataset['type'] == 'Duke':
                    parts = filename.split('_')
                    if len(parts) == 3:
                        if parts[0] in picked_ids:
                            shutil.copy(original_path, dest_path) 
                        else:
                            shutil.copy(transferred_path, dest_path) 
                    else:
                        print(f"Skipping file {filename} due to unexpected format")
                        continue
    