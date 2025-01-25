import os
import sys
sys.path.append('/home/jun/stargan')
import argparse
from solver import Solver
from solver_ver2 import Solver_ver2
from data_loader import get_loader, get_re_id_loader
from torch.backends import cudnn
from data_loader import CelebA
from torchvision import transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import re
import torchvision.transforms as transforms
def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()
# Model configuration.
parser.add_argument('--c_dim', type=int, default=2, help='dimension of domain labels (1st dataset)')
parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
parser.add_argument('--reid_crop_size', type=int, default=256, help='crop size for the PersonReID dataset')
parser.add_argument('--image_size', type=int, default=286, help='image resolution')
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_idt', type=float, default=0.5, help='weight for segmentation loss')
parser.add_argument('--lambda_seg', type=float, default=0.5, help='weight for segmentation loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

# Training configuration.
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both', 'PersonReID'])
parser.add_argument('--parsing', action='store_true')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
parser.add_argument('--cam', type=str, default='multi', choices=['multi', 'single']) 
# Test configuration.
parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

# Miscellaneous.
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--use_tensorboard', type=str2bool, default=True)

# Directories.
parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
parser.add_argument('--reid_image_dir', type=str, default='/home/jun/ReID_Dataset')
parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
parser.add_argument('--log_dir', type=str, default='test/logs')
parser.add_argument('--model_save_dir', type=str, default='test/models')
parser.add_argument('--sample_dir', type=str, default='test/samples')
parser.add_argument('--result_dir', type=str, default='test/results')

# Step size.
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=10000)
parser.add_argument('--lr_update_step', type=int, default=1000)

config = parser.parse_args()


# For fast training.
cudnn.benchmark = True
celeba_loader = None
rafd_loader = None
personreid_loader = get_re_id_loader(config.reid_image_dir, 
                                 config.reid_crop_size, config.image_size, config.batch_size,
                                 config.mode, config.cam, config.num_workers) 

# Function to convert tensor to image for displaying
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu()  # Clone the tensor to avoid modifying the original and move to CPU
    tensor = F.interpolate(tensor, size=[256,128], mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)  # De-normalize
    tensor = tensor.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    tensor = tensor.numpy()  # Convert to numpy array
    tensor = (tensor * 255).astype(np.uint8)  # Convert to uint8
    return tensor 

crop_size_w = 256
crop_size_h = 128

transform = transforms.Compose([
    transforms.Resize((crop_size_h, crop_size_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

  
raw_data_dir_market1501 = Path('/home/jun/ReID_Dataset/DukeMTMC-reID')
# Make dictionary of all image market1501 with keys of image_name and values of image_path
market_train_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('bounding_box_train/*.jpg')))}
market_gallery_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('bounding_box_test/*.jpg')))}
market_query_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('query/*.jpg')))}

cam_num = 1

list = ['singlecam_0_0','singlecam_0_5','singlecam_1_0','singlecam_1_5','singlecam_2_0','singlecam_2_5']
for name in list:
    solver = Solver_ver2(celeba_loader, rafd_loader, personreid_loader, config)
    #solver.G.load_state_dict(torch.load(f'/home/jun/stargan/checkpoints/personreid_parsing_{name}/models/200000-G.ckpt'))
    solver.G.load_state_dict(torch.load(f'/home/jun/stargan/checkpoints/personreid_spgan_{name}/models/200000-G.ckpt'))
    solver.G.to(solver.device)
        
    #new_dataset_parrent_dir = Path(f'/home/jun/ReID_Dataset/DukeMTMC-reID-stargan-{name}')
    new_dataset_parrent_dir = Path(f'/home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-{name}')
    new_dataset_parrent_dir.mkdir(exist_ok=True)
    for i in range(cam_num):
        new_data_dir_market1501 = new_dataset_parrent_dir/ f'DukeMTMC-reID-stargan-cam{i}'
        new_data_dir_market1501.mkdir(exist_ok=True)
        new_market_train_dir = new_data_dir_market1501 / 'bounding_box_train'
        new_market_train_dir.mkdir(exist_ok=True)
        new_market_gallery_dir = new_data_dir_market1501 / 'bounding_box_test'
        new_market_gallery_dir.mkdir(exist_ok=True)
        new_market_query_dir = new_data_dir_market1501 / 'query'
        new_market_query_dir.mkdir(exist_ok=True)

        print(f'Making {str(new_data_dir_market1501)}')
        
        for image_name, image_path in market_train_dic.items():
            image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
            # Apply the transformations
            transformed_image = transform(image)
            x = transformed_image.unsqueeze(0).to(solver.device)
            camid_one_hot_vector = torch.zeros(config.c_dim)
            camid_one_hot_vector[i] = 1 
            label = camid_one_hot_vector.unsqueeze(0).to(solver.device)
            with torch.no_grad():
                y = solver.G(x,label)
            output_image = tensor_to_image(y)
            output_image = Image.fromarray(output_image)
            output_image_path = new_market_train_dir / image_name
            output_image.save(output_image_path)