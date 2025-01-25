from pathlib import Path
import os
from tqdm.auto import tqdm
import argparse
import scipy.io
import json
parser = argparse.ArgumentParser(description='Prepare dataset for market1501, dukemtmc, cuhk03')
parser.add_argument('--market1501', action='store_true', help='Prepare market1501 dataset' )
parser.add_argument('--cuhk03', action='store_true', help='Prepare cuhk03 dataset' )
parser.add_argument('--dukemtmc', action='store_true', help='Prepare market1501 dataset' )
parser.add_argument('--mat', action='store_true', help='Prepare cuhk03 dataset with mat file' )
parser.add_argument('--json', action='store_true', help='Prepare cuhk03 dataset with json file' )
opt = parser.parse_args()

if opt.market1501:
    # Import library
    from pathlib import Path

    # Market1501 dataset dir
    raw_data_dir_market1501 = Path('/home/jun/ReID_Dataset/market_SinglePic_delete_background/')

    # Make dictionary of all image market1501 with keys of image_name and values of image_path
    market_train_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('bounding_box_train/*.jpg')))}
    market_gallery_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('bounding_box_test/*.jpg')))}
    market_query_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('query/*.jpg')))}

    # Make new dataset dir
    new_market_dir = raw_data_dir_market1501 / 'dataloader' 
    new_market_dir.mkdir(exist_ok=True) # Making new dataset dir as olod_dir/dataloader
    market_train_dir = new_market_dir / 'train'
    market_gallery_dir = new_market_dir / 'gallery'
    market_query_dir = new_market_dir / 'query'
    market_train_dir.mkdir(exist_ok=True) # Making train dir
    market_gallery_dir.mkdir(exist_ok=True) # Making gallery dir
    market_query_dir.mkdir(exist_ok=True) # Making query dir

    # Making train and val dataset (first picture will be use for validation)
    market_train_val_dir = new_market_dir / 'train_val'
    market_train_val_dir.mkdir(exist_ok=True)
    market_train_small_dir = market_train_val_dir / 'train'
    market_train_small_dir.mkdir(exist_ok=True) 
    market_val_dir = market_train_val_dir / 'val'
    market_val_dir.mkdir(exist_ok=True) 
    # Proceed train and val data
    print('Proceeding train and val data')
    for img_name, img_path in tqdm(market_train_dic.items()):
        label = img_name.split('_')[0]
        train_label = market_train_dir / label # Create train/<label> folder
        train_label.mkdir(exist_ok=True)
        train_small_label = market_train_small_dir / label # Create train_val/train/<label> folder
        train_small_label.mkdir(exist_ok=True)
        val_label = market_val_dir / label # Create train_val/val/<label> folder
        val_label.mkdir(exist_ok=True)
        (train_label / img_name).write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (train/<label>/*.img)
        if len(list(val_label.glob('*.jpg'))) < 1:
            (val_label / img_name).write_bytes(img_path.read_bytes()) # Put first pic of each label into train_val/val/<label>/*.img
        else:
            (train_small_label / img_name).write_bytes(img_path.read_bytes()) # Put every pic except fist one of each label into train_val/train/<label>/*.img 
        
    # Process gallery data     
    print('Proceeding gallery data')
    for i, (img_name, img_path) in tqdm(enumerate(market_gallery_dic.items())):
        label = img_name.split('_')[0]
        gallery_label = market_gallery_dir / label # Create gallery/<label> folder
        gallery_label.mkdir(exist_ok=True)
        (gallery_label / img_name).write_bytes(img_path.read_bytes()) # Put all image to gallery folder based on it label (gallery/<label>/*.img)
        
    # Proceed query data
    print('Proceeding query data')
    for i, (img_name, img_path) in tqdm(enumerate(market_query_dic.items())):
        label = img_name.split('_')[0]
        query_label = market_query_dir / label # Create query/<label> folder
        query_label.mkdir(exist_ok=True)
        (query_label / img_name).write_bytes(img_path.read_bytes()) # Put all image to query folder based on it label (query/<label>/*.img)

if opt.dukemtmc:
    # DukeMTMC dataset dir
    raw_data_dir_duke = Path('/home/jun/ReID_Dataset/duke_SinglePic_delete_background/')

    # Make dictionary of all image DukeMTMC with keys of image_name and values of image_path
    duke_train_dic = {i.name:i for i in sorted(list(raw_data_dir_duke.glob('bounding_box_train/*.jpg')))}
    duke_gallery_dic = {i.name:i for i in sorted(list(raw_data_dir_duke.glob('bounding_box_test/*.jpg')))}
    duke_query_dic = {i.name:i for i in sorted(list(raw_data_dir_duke.glob('query/*.jpg')))}

    # Make new dataset dir
    new_duke_dir = raw_data_dir_duke / 'dataloader' 
    new_duke_dir.mkdir(exist_ok=True) # Making new dataset dir as olod_dir/dataloader
    duke_train_dir = new_duke_dir / 'train'
    duke_gallery_dir = new_duke_dir / 'gallery'
    duke_query_dir = new_duke_dir / 'query'
    duke_train_dir.mkdir(exist_ok=True) # Making train dir
    duke_gallery_dir.mkdir(exist_ok=True) # Making gallery dir
    duke_query_dir.mkdir(exist_ok=True) # Making query dir

    # Making train and val dataset (first picture will be use for validation)
    duke_train_val_dir = new_duke_dir / 'train_val'
    duke_train_val_dir.mkdir(exist_ok=True)
    duke_train_small_dir = duke_train_val_dir / 'train'
    duke_train_small_dir.mkdir(exist_ok=True) 
    duke_val_dir = duke_train_val_dir / 'val'
    duke_val_dir.mkdir(exist_ok=True) 
    
    # Proceed train and val data
    print('Proceeding train and val data')
    for img_name, img_path in tqdm(duke_train_dic.items()):
        label = img_name.split('_')[0]
        train_label = duke_train_dir / label # Create train/<label> folder
        train_label.mkdir(exist_ok=True)
        train_small_label = duke_train_small_dir / label # Create train_val/train/<label> folder
        train_small_label.mkdir(exist_ok=True)
        val_label = duke_val_dir / label # Create train_val/val/<label> folder
        val_label.mkdir(exist_ok=True)
        (train_label / img_name).write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (train/<label>/*.img)
        if len(list(val_label.glob('*.jpg'))) < 1:
            (val_label / img_name).write_bytes(img_path.read_bytes()) # Put first pic of each label into train_val/val/<label>/*.img
        else:
            (train_small_label / img_name).write_bytes(img_path.read_bytes()) # Put every pic except fist one of each label into train_val/train/<label>/*.img 
        
    # Process gallery data     
    print('Proceeding gallery data')
    for img_name, img_path in tqdm(duke_gallery_dic.items()):
        label = img_name.split('_')[0]
        gallery_label = duke_gallery_dir / label # Create gallery/<label> folder
        gallery_label.mkdir(exist_ok=True)
        (gallery_label / img_name).write_bytes(img_path.read_bytes()) # Put all image to gallery folder based on it label (gallery/<label>/*.img)
        
    # Proceed query data
    print('Proceeding query data')
    for img_name, img_path in tqdm(duke_query_dic.items()):
        label = img_name.split('_')[0]
        query_label = duke_query_dir / label # Create query/<label> folder
        query_label.mkdir(exist_ok=True)
        (query_label / img_name).write_bytes(img_path.read_bytes()) # Put all image to query folder based on it label (query/<label>/*.img)

if opt.cuhk03:
    if opt.mat:
        detected = scipy.io.loadmat('/home/jun/ReID_Dataset/cuhk03/cuhk03_new_protocol_config_detected.mat')
        labeled = scipy.io.loadmat('/home/jun/ReID_Dataset/cuhk03/cuhk03_new_protocol_config_labeled.mat')
        # cuhk03detected dataset dir
        raw_data_dir_cuhk03 = Path('/home/jun/ReID_Dataset/cuhk03')
        detected = scipy.io.loadmat('/home/jun/ReID_Dataset/cuhk03/cuhk03_new_protocol_config_detected.mat')

        # Make dictionary of all image cuhk03detected with keys of image_name and values of image_path
        detected_dic = {i.name:i for i in sorted(list(raw_data_dir_cuhk03.glob('images_detected/*.png')))}

        # Make new dataset dir for labeled data
        new_cuhk03detected_dir = raw_data_dir_cuhk03 / 'dataloader_new_protocal_detected' 
        new_cuhk03detected_dir.mkdir(exist_ok=True) # Making new dataset dir as olod_dir/dataloader
        cuhk03detected_train_dir = new_cuhk03detected_dir / 'train'
        cuhk03detected_gallery_dir = new_cuhk03detected_dir / 'gallery'
        cuhk03detected_query_dir = new_cuhk03detected_dir / 'query'
        cuhk03detected_train_dir.mkdir(exist_ok=True) # Making train dir
        cuhk03detected_gallery_dir.mkdir(exist_ok=True) # Making gallery dir
        cuhk03detected_query_dir.mkdir(exist_ok=True) # Making query dir

        # Making train and val dataset (first picture will be use for validation)
        cuhk03detected_train_val_dir = new_cuhk03detected_dir / 'train_val'
        cuhk03detected_train_val_dir.mkdir(exist_ok=True)
        cuhk03detected_train_small_dir = cuhk03detected_train_val_dir / 'train'
        cuhk03detected_train_small_dir.mkdir(exist_ok=True) 
        cuhk03detected_val_dir = cuhk03detected_train_val_dir / 'val'
        cuhk03detected_val_dir.mkdir(exist_ok=True)

        for id in tqdm(range(len(detected['filelist']))): # Get image id
            img_label = int(detected['labels'][id].squeeze(axis=0)) # Get label of each image
            img_name = str(detected['filelist'].squeeze()[id][0]) # Get name of each image
            
            img_path = detected_dic[img_name]
            
            # Train, train_val data
            if img_label in list(detected['train_idx'].squeeze()): # If image id is in train_idx, put it in train folder
                train_label_folder_path = cuhk03detected_train_dir / str(img_label) # Make train label folder (train/<label>)
                train_label_folder_path.mkdir(exist_ok=True) 
                small_train_label_folder_path = cuhk03detected_train_small_dir / str(img_label) # Make train_val/train/<label> folder
                small_train_label_folder_path.mkdir(exist_ok=True) 
                val_label_folder_path = cuhk03detected_val_dir / str(img_label) # Make train_val/val/<label> folder
                val_label_folder_path.mkdir(exist_ok=True) 
                (train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (train/<label>/*.img)
            if len(list(val_label_folder_path.glob('*.jpg'))) < 1:
                (val_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put first pic of each label into train_val/val/<label>/*.img
            else:
                (small_train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put every pic except fist one of each label into train_val/train/<label>/*.img   

            # Test data
            if img_label in list(detected['gallery_idx'].squeeze()): # If image id is in gallery_idx , put it in train folder
                gallery_label_folder_path = cuhk03detected_gallery_dir / str(img_label) # Make gallery label folder (gallery/<label>)
                gallery_label_folder_path.mkdir(exist_ok=True)  
                target_img_path = gallery_label_folder_path / img_name
                target_img_path.write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (gallery/<label>/*.img)
                
            # Query data
            if img_label in list(detected['query_idx'].squeeze()): # If image id is in query_idx , put it in query folder
                query_label_folder_path = cuhk03detected_query_dir / str(img_label) # Make query label folder (query/<label>)
                query_label_folder_path.mkdir(exist_ok=True)   
                target_img_path = query_label_folder_path / img_name
                target_img_path.write_bytes(img_path.read_bytes()) # Put all image to query folder based on it label (query/<label>/*.img)
                
        # cuhk03labeled dataset dir
        raw_data_dir_cuhk03 = Path('/home/jun/ReID_Dataset/cuhk03')
        labeled = scipy.io.loadmat('/home/jun/ReID_Dataset/cuhk03/cuhk03_new_protocol_config_labeled.mat')

        # Make dictionary of all image cuhk03labeled with keys of image_name and values of image_path
        labeled_dic = {i.name:i for i in sorted(list(raw_data_dir_cuhk03.glob('images_labeled/*.png')))}

        # Make new dataset dir for labeled data
        new_cuhk03labeled_dir = raw_data_dir_cuhk03 / 'dataloader_new_protocol_labeled' 
        new_cuhk03labeled_dir.mkdir(exist_ok=True) # Making new dataset dir as olod_dir/dataloader
        cuhk03labeled_train_dir = new_cuhk03labeled_dir / 'train'
        cuhk03labeled_gallery_dir = new_cuhk03labeled_dir / 'gallery'
        cuhk03labeled_query_dir = new_cuhk03labeled_dir / 'query'
        cuhk03labeled_train_dir.mkdir(exist_ok=True) # Making train dir
        cuhk03labeled_gallery_dir.mkdir(exist_ok=True) # Making gallery dir
        cuhk03labeled_query_dir.mkdir(exist_ok=True) # Making query dir

        # Making train and val dataset (first picture will be use for validation)
        cuhk03labeled_train_val_dir = new_cuhk03labeled_dir / 'train_val'
        cuhk03labeled_train_val_dir.mkdir(exist_ok=True)
        cuhk03labeled_train_small_dir = cuhk03labeled_train_val_dir / 'train'
        cuhk03labeled_train_small_dir.mkdir(exist_ok=True) 
        cuhk03labeled_val_dir = cuhk03labeled_train_val_dir / 'val'
        cuhk03labeled_val_dir.mkdir(exist_ok=True)

        for id in tqdm(range(len(labeled['filelist']))): # Get image id
            img_label = int(labeled['labels'][id].squeeze(axis=0)) # Get label of each image
            img_name = str(labeled['filelist'].squeeze()[id][0]) # Get name of each image
        
            img_path = labeled_dic[img_name]

            
            # Train, train_val data
            if img_label in list(labeled['train_idx'].squeeze()): # If image id is in train_idx, put it in train folder
                train_label_folder_path = cuhk03labeled_train_dir / str(img_label) # Make train label folder (train/<label>)
                train_label_folder_path.mkdir(exist_ok=True) 
                small_train_label_folder_path = cuhk03labeled_train_small_dir / str(img_label) # Make train_val/train/<label> folder
                small_train_label_folder_path.mkdir(exist_ok=True) 
                val_label_folder_path = cuhk03labeled_val_dir / str(img_label) # Make train_val/val/<label> folder
                val_label_folder_path.mkdir(exist_ok=True) 
                (train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (train/<label>/*.img)
            if len(list(val_label_folder_path.glob('*.jpg'))) < 1:
                (val_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put first pic of each label into train_val/val/<label>/*.img
            else:
                (small_train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put every pic except fist one of each label into train_val/train/<label>/*.img   

            # Test data
            if img_label in list(labeled['gallery_idx'].squeeze()): # If image id is in gallery_idx , put it in train folder
                gallery_label_folder_path = cuhk03labeled_gallery_dir / str(img_label) # Make gallery label folder (gallery/<label>)
                gallery_label_folder_path.mkdir(exist_ok=True)  
                target_img_path = gallery_label_folder_path / img_name
                target_img_path.write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (gallery/<label>/*.img)
                
            # Query data
            if img_label in list(labeled['query_idx'].squeeze()): # If image id is in query_idx , put it in query folder
                query_label_folder_path = cuhk03labeled_query_dir / str(img_label) # Make query label folder (query/<label>)
                query_label_folder_path.mkdir(exist_ok=True)   
                target_img_path = query_label_folder_path / img_name
                target_img_path.write_bytes(img_path.read_bytes()) # Put all image to query folder based on it label (query/<label>/*.img)
    if opt.json:
        # cuhk03detected dataset dir
        raw_data_dir_cuhk03 = Path('/home/jun/ReID_Dataset/cuhk03_SinglePic_delete_background/')

        # Read split config file
        f = open('/home/jun/ReID_Dataset/cuhk03_PI/splits_classic_detected.json')
        file = json.load(f)

        # Set split id (total of 20):
        split_num = 0

        # Train, gallery, query file list: 
        train_list = file[split_num]['train']
        gallery_list = file[split_num]['gallery']
        query_list = file[split_num]['query']

        # Make dictionary of all image dataset with keys of image_name and values of image_path

        detected_dic = {i.name:i for i in sorted(list(raw_data_dir_cuhk03.glob('images_detected/*.png')))}

        # Make new dataset dir for labeled data
        new_dataset_dir = raw_data_dir_cuhk03 / 'dataloader_classic_detected' 
        new_dataset_dir.mkdir(exist_ok=True) # Making new dataset dir as olod_dir/dataloader
        dataset_train_dir = new_dataset_dir / 'train'
        dataset_gallery_dir = new_dataset_dir / 'gallery'
        dataset_query_dir = new_dataset_dir / 'query'
        dataset_train_dir.mkdir(exist_ok=True) # Making train dir
        dataset_gallery_dir.mkdir(exist_ok=True) # Making gallery dir
        dataset_query_dir.mkdir(exist_ok=True) # Making query dir

        # Making train and val dataset (first picture will be use for validation)
        dataset_train_val_dir = new_dataset_dir / 'train_val'
        dataset_train_val_dir.mkdir(exist_ok=True)
        dataset_train_small_dir = dataset_train_val_dir / 'train'
        dataset_train_small_dir.mkdir(exist_ok=True) 
        dataset_val_dir = dataset_train_val_dir / 'val'
        dataset_val_dir.mkdir(exist_ok=True)


        # Making train dataset:
        for img_path, img_label, camera_id in tqdm(train_list): # file[split_num]['train'] = [['file_path', <Image label>, <Camera ID>], [...], [...]]]
            img_path = Path(img_path.replace('\\','/').replace('./data','/home/jun/ReID_Dataset'))
            img_name = img_path.name
            train_label_folder_path = dataset_train_dir / str(img_label) # Make train label folder (train/<label>)
            train_label_folder_path.mkdir(exist_ok=True) 
            small_train_label_folder_path = dataset_train_small_dir / str(img_label) # Make train_val/train/<label> folder
            small_train_label_folder_path.mkdir(exist_ok=True) 
            val_label_folder_path = dataset_val_dir / str(img_label) # Make train_val/val/<label> folder
            val_label_folder_path.mkdir(exist_ok=True)
            (train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (train/<label>/*.img)
            if len(list(val_label_folder_path.glob('*.png'))) < 1:
                (val_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put first pic of each label into train_val/val/<label>/*.img
            else:
                (small_train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put every pic except fist one of each label into train_val/train/<label>/*.img   
        # Making gallery dataset:
        for img_path, img_label, camera_id in tqdm(gallery_list): # file[split_num]['train'] = [['file_path', <Image label>, <Camera ID>], [...], [...]]]
            img_path = Path(img_path.replace('\\','/').replace('./data','/home/jun/ReID_Dataset'))
            img_name = img_path.name
            gallery_label_folder_path = dataset_gallery_dir / str(img_label) # Make gallery label folder (gallery/<label>)
            gallery_label_folder_path.mkdir(exist_ok=True)  
            target_img_path = gallery_label_folder_path / img_name
            target_img_path.write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (gallery/<label>/*.img)
            
        # Making query dataset:
        for img_path, img_label, camera_id in tqdm(query_list): # file[split_num]['train'] = [['file_path', <Image label>, <Camera ID>], [...], [...]]]
            img_path = Path(img_path.replace('\\','/').replace('./data','/home/jun/ReID_Dataset'))
            img_name = img_path.name
            query_label_folder_path = dataset_query_dir / str(img_label) # Make query label folder (query/<label>)
            query_label_folder_path.mkdir(exist_ok=True)   
            target_img_path = query_label_folder_path / img_name
            target_img_path.write_bytes(img_path.read_bytes()) # Put all image to query folder based on it label (query/<label>/*.img)

            
        # Read split config file
        f = open('/home/jun/ReID_Dataset/cuhk03/splits_new_detected.json')
        file = json.load(f)

        # Set split id (total of 20):
        split_num = 0

        # Train, gallery, query file list: 
        train_list = file[split_num]['train']
        gallery_list = file[split_num]['gallery']
        query_list = file[split_num]['query']

        # Make dictionary of all image dataset with keys of image_name and values of image_path

        detected_dic = {i.name:i for i in sorted(list(raw_data_dir_cuhk03.glob('images_detected/*.png')))}

        # Make new dataset dir for labeled data
        new_dataset_dir = raw_data_dir_cuhk03 / 'dataloader_new_detected' 
        new_dataset_dir.mkdir(exist_ok=True) # Making new dataset dir as olod_dir/dataloader
        dataset_train_dir = new_dataset_dir / 'train'
        dataset_gallery_dir = new_dataset_dir / 'gallery'
        dataset_query_dir = new_dataset_dir / 'query'
        dataset_train_dir.mkdir(exist_ok=True) # Making train dir
        dataset_gallery_dir.mkdir(exist_ok=True) # Making gallery dir
        dataset_query_dir.mkdir(exist_ok=True) # Making query dir

        # Making train and val dataset (first picture will be use for validation)
        dataset_train_val_dir = new_dataset_dir / 'train_val'
        dataset_train_val_dir.mkdir(exist_ok=True)
        dataset_train_small_dir = dataset_train_val_dir / 'train'
        dataset_train_small_dir.mkdir(exist_ok=True) 
        dataset_val_dir = dataset_train_val_dir / 'val'
        dataset_val_dir.mkdir(exist_ok=True)


        # Making train dataset:
        for img_path, img_label, camera_id in tqdm(train_list): # file[split_num]['train'] = [['file_path', <Image label>, <Camera ID>], [...], [...]]]
            img_path = Path(img_path.replace('\\','/').replace('./data','/home/jun/ReID_Dataset'))
            img_name = img_path.name
            train_label_folder_path = dataset_train_dir / str(img_label) # Make train label folder (train/<label>)
            train_label_folder_path.mkdir(exist_ok=True) 
            small_train_label_folder_path = dataset_train_small_dir / str(img_label) # Make train_val/train/<label> folder
            small_train_label_folder_path.mkdir(exist_ok=True) 
            val_label_folder_path = dataset_val_dir / str(img_label) # Make train_val/val/<label> folder
            val_label_folder_path.mkdir(exist_ok=True)
            (train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (train/<label>/*.img)
            if len(list(val_label_folder_path.glob('*.png'))) < 1:
                (val_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put first pic of each label into train_val/val/<label>/*.img
            else:
                (small_train_label_folder_path / img_name).write_bytes(img_path.read_bytes()) # Put every pic except fist one of each label into train_val/train/<label>/*.img   

        # Making gallery dataset:
        for img_path, img_label, camera_id in tqdm(gallery_list): # file[split_num]['train'] = [['file_path', <Image label>, <Camera ID>], [...], [...]]]
            img_path = Path(img_path.replace('\\','/').replace('./data','/home/jun/ReID_Dataset'))
            img_name = img_path.name
            gallery_label_folder_path = dataset_gallery_dir / str(img_label) # Make gallery label folder (gallery/<label>)
            gallery_label_folder_path.mkdir(exist_ok=True)  
            target_img_path = gallery_label_folder_path / img_name
            target_img_path.write_bytes(img_path.read_bytes()) # Put all image to train folder based on it label (gallery/<label>/*.img)
            
        # Making query dataset:
        for img_path, img_label, camera_id in tqdm(query_list): # file[split_num]['train'] = [['file_path', <Image label>, <Camera ID>], [...], [...]]]
            img_path = Path(img_path.replace('\\','/').replace('./data','/home/jun/ReID_Dataset'))
            img_name = img_path.name
            query_label_folder_path = dataset_query_dir / str(img_label) # Make query label folder (query/<label>)
            query_label_folder_path.mkdir(exist_ok=True)   
            target_img_path = query_label_folder_path / img_name
            target_img_path.write_bytes(img_path.read_bytes()) # Put all image to query folder based on it label (query/<label>/*.img)
