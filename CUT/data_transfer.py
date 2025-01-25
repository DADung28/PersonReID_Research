import os
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import argparse
import os
from util import util
import torch
import models
import data
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from options.transfer_options import TransferOptions
from tqdm import tqdm


if __name__ == '__main__':
    opt = TransferOptions().parse()   # get training options
    model = create_model(opt)      # create a model given opt.model and other options
    model.load_networks('latest') 
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

# Dataset loader
transform = transforms.Compose([
    transforms.Resize((opt.crop_size, int(opt.crop_size/2))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Market1501 dataset dir
raw_data_dir_market1501 = Path(opt.source_dir)
# Make dictionary of all image market1501 with keys of image_name and values of image_path
market_train_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('bounding_box_train/*.jpg')))}
market_gallery_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('bounding_box_test/*.jpg')))}
market_query_dic = {i.name:i for i in sorted(list(raw_data_dir_market1501.glob('query/*.jpg')))}

new_data_dir_market1501 = Path(opt.save_dir)
new_data_dir_market1501.mkdir(exist_ok=True)
new_market_train_dir = new_data_dir_market1501 / 'bounding_box_train'
new_market_train_dir.mkdir(exist_ok=True)
new_market_gallery_dir = new_data_dir_market1501 / 'bounding_box_test'
new_market_gallery_dir.mkdir(exist_ok=True)
new_market_query_dir = new_data_dir_market1501 / 'query'
new_market_query_dir.mkdir(exist_ok=True)


for image_name, image_path in tqdm(market_train_dic.items()):
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image)
    x = transformed_image.unsqueeze(0).cuda()
    data = {'A':x,'B':x,
            'A_paths': image_path,
            'B_paths': image_path}
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    # Assuming `data` is the dictionary containing the tensors

    output_image = tensor_to_image(visuals['fake_B'])
    output_image = Image.fromarray(output_image)
    output_image_path = new_market_train_dir / image_name
    output_image.save(output_image_path)