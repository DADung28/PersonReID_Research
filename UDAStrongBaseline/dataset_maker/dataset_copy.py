import os
import shutil

dataset_dir = '/home/jun/ReID_Dataset'

name_list = ['0_0','0_1','0_2','0_5','1_0','1_1','1_2','1_5','2_0','2_1','2_2','2_5','5_0','5_1','5_2','5_5','singlecam']

for dataset_name in ['Market-1501-v15.09.15','DukeMTMC-reID']:
    source_dataset = os.path.join(dataset_dir,dataset_name)
    for i in name_list:
        target_dataset = os.path.join(dataset_dir, dataset_name+'-stargan-'+i, dataset_name)
        
        print(f'Copying {source_dataset} to {target_dataset}')
        try:
            shutil.copytree(source_dataset,target_dataset, dirs_exist_ok=False)
        except:
            pass 
        
        for cam_set in ['randomcam','allcam']:
            random_cam = target_dataset = os.path.join(dataset_dir, dataset_name+'-stargan-'+i, dataset_name+'-stargan-'+cam_set)
            
            source_test = os.path.join(source_dataset, 'bounding_box_test')
            target_test = os.path.join(random_cam, 'bounding_box_test')
            source_query = os.path.join(source_dataset, 'query')
            target_query = os.path.join(random_cam, 'query')
            
            print(f'Copying {source_test} to {target_test}')
            
            try:
                shutil.copytree(source_test,target_test, dirs_exist_ok=False) 
            except:
                pass 

            print(f'Copying {source_query} to {target_query}')
            
            try:
                shutil.copytree(source_query,target_query, dirs_exist_ok=False) 
            except:
                pass 