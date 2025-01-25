import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
import numpy as np
import os

# Define a color palette for the segmentation map classes (e.g., 6 classes)
color_palette = [
    (0, 0, 0),        # Class 0: Black
    (255, 0, 0),      # Class 1: Red
    (0, 255, 0),      # Class 2: Green
    (0, 0, 255),      # Class 3: Blue
    (255, 255, 0),    # Class 4: Yellow
    (255, 0, 255)     # Class 5: Magenta
]

def apply_color_map(segmap, palette):
    """
    Apply a color map to a segmentation map.
    :param segmap: Segmentation map (H, W) with class indices.
    :param palette: Color palette as a list of RGB tuples.
    :return: Color-mapped image (H, W, 3).
    """
    h, w = segmap.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        color_image[segmap == class_idx] = color
    return color_image

def tensor_to_pil(tensor):
    """
    Convert a normalized tensor to a PIL image.
    :param tensor: Normalized tensor (C, H, W).
    :return: PIL image.
    """
    tensor = tensor.squeeze(0)  # Remove batch dimension if exists
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()
    tensor = (tensor * 0.5 + 0.5) * 255  # Assuming tensor was normalized to [-1, 1]
    tensor = tensor.astype(np.uint8)
    return Image.fromarray(tensor)

def save_segmentation_maps(a_real_img, b_fake_img, 
                           a_real_segmap, b_fake_segmap, 
                           save_dir, epoch, i):
    # Ensure the segmentation maps are on the CPU and convert them to NumPy arrays
    a_real_np = a_real_segmap.detach().cpu().numpy()
    b_fake_np = b_fake_segmap.detach().cpu().numpy()

    # Apply color mapping to the segmentation maps
    a_real_color = apply_color_map(a_real_np[0], color_palette)
    b_fake_color = apply_color_map(b_fake_np[0], color_palette)

    # Convert the color-mapped arrays to PIL images
    a_real_seg_img = Image.fromarray(a_real_color)
    b_fake_seg_img = Image.fromarray(b_fake_color)

    # Denormalize and convert original images to PIL images
    mean = [0.5] * 3
    std = [0.5] * 3
    #a_real_pil_img = Image.fromarray((denormalize(a_real_img.squeeze(0).detach().cpu().clone(), mean, std).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    #b_fake_pil_img = Image.fromarray((denormalize(b_fake_img.squeeze(0).detach().cpu().clone(), mean, std).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    a_real_pil_img = tensor_to_pil(a_real_img)
    b_fake_pil_img = tensor_to_pil(b_fake_img)
   
    
    target_size = (128, 256)
    a_real_seg_img = a_real_seg_img.resize(target_size)
    b_fake_seg_img = b_fake_seg_img.resize(target_size)
    a_real_pil_img = a_real_pil_img.resize(target_size)
    b_fake_pil_img = b_fake_pil_img.resize(target_size)
    
    # Combine original images with their corresponding segmentation maps vertically
    a_real_combined = Image.new('RGB', (a_real_pil_img.width, a_real_pil_img.height + a_real_seg_img.height))
    b_fake_combined = Image.new('RGB', (b_fake_pil_img.width, b_fake_pil_img.height + b_fake_seg_img.height))

    
    a_real_combined.paste(a_real_pil_img, (0, 0))
    a_real_combined.paste(a_real_seg_img, (0, a_real_pil_img.height))
    b_fake_combined.paste(b_fake_pil_img, (0, 0))
    b_fake_combined.paste(b_fake_seg_img, (0, b_fake_pil_img.height))

    # Merge combined images horizontally
    merged_width = a_real_combined.width + b_fake_combined.width 
    merged_image = Image.new('RGB', (merged_width, a_real_combined.height))
    merged_image.paste(a_real_combined, (0, 0))
    merged_image.paste(b_fake_combined, (a_real_combined.width, 0))
   
    # Create the file name
    file_name = '%s/Epoch_(%d)_segmap_(%d).jpg' % (save_dir, epoch, i + 1)
    
    # Save the image
    merged_image.save(file_name)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    image_save_dir = opt.image_save_dir 
    os.makedirs(image_save_dir, exist_ok=True)
    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                
                a_real = model.real_A 
                a_real_segmap = torch.argmax(model.real_A_segmap, dim = 1)
                b_fake = model.fake_B 
                b_fake_segmap = torch.argmax(model.fake_B_segmap, dim = 1)
                save_segmentation_maps(a_real, b_fake, a_real_segmap, b_fake_segmap, image_save_dir, epoch, i)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
