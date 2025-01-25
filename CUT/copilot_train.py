import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
import numpy as np
import os
from UDAsbs.utils.data import transforms as T
from UDAsbs import datasets
import os.path as osp
from UDAsbs.utils.data.sampler import RandomMultipleGallerySampler
from torch.utils.data import DataLoader
from UDAsbs.utils.data.preprocessor import Preprocessor
from UDAsbs.utils.data import IterLoader
# Define a color palette for the segmentation map classes (e.g., 6 classes)

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

def save_img(real_images, fake_images, save_dir, epoch, i):
    # Convert real_images and fake_images to PIL images
    real_images = [tensor_to_pil(img) for img in real_images]
    fake_images = [tensor_to_pil(img) for img in fake_images]

    # Resize images to the same size (assuming all images have the same size)
    target_size = (128, 256)
    real_images_resized = [img.resize(target_size) for img in real_images]
    fake_images_resized = [img.resize(target_size) for img in fake_images]

    # Create a blank image with the appropriate dimensions
    total_width = target_size[0] * len(real_images)
    total_height = target_size[1] * 2  # Two rows (real images in the first row, fake images in the second row)
    merged_image = Image.new('RGB', (total_width, total_height))

    # Paste real images in the first row
    x_offset = 0
    for img in real_images_resized:
        merged_image.paste(img, (x_offset, 0))
        x_offset += target_size[0]

    # Paste fake images in the second row
    x_offset = 0
    for img in fake_images_resized:
        merged_image.paste(img, (x_offset, target_size[1]))
        x_offset += target_size[0]

    # Save the merged image
    save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{i}.png')
    merged_image.save(save_path)


def get_data(data_dir, load_size, crop_size, batch_size=4, workers=16, num_instances=2, iters=3000):
    A = 'market1501'
    B = 'dukemtmc'
    root = osp.join(data_dir)
    dataset_A = datasets.create(A, root)
    train_A = dataset_A.train
    dataset_B = datasets.create(B, root)
    train_B = dataset_B.train

    train_transformer = T.Compose([
            T.Resize((load_size,load_size), Image.BICUBIC),
            T.RandomCrop((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
            #T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])


    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler_A = RandomMultipleGallerySampler(train_A, num_instances)
        sampler_B = RandomMultipleGallerySampler(train_B, num_instances)
        #sample = None
    else:
        sampler = None

    A_loader = IterLoader(DataLoader(Preprocessor(train_A, root=dataset_A.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler_A,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True))

    B_loader = IterLoader(DataLoader(Preprocessor(train_B, root=dataset_B.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler_B,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True))

    return A_loader, B_loader


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    image_save_dir = opt.image_save_dir 
    os.makedirs(image_save_dir, exist_ok=True)
    model = create_model(opt)      # create a model given opt.model and other options
    #print('The number of training images = %d' % dataset_size)
    iters = 3000
    num_instances = 2
    datasetA, datasetB = get_data(opt.dataroot, opt.load_size, opt.crop_size, batch_size=4, workers=16, num_instances=num_instances, iters=iters)
   
    
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
        
        for i in range(iters):  # inner loop within one epoch
            data = {}
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            data['A'] = datasetA.next()[0]
            data['A_paths'] = datasetA.next()[1]
            data['A_id'] = datasetA.next()[2]
            
            data['B'] = datasetB.next()[0] 
            data['B_paths'] = datasetB.next()[1] 
            data['B_id'] = datasetB.next()[2] 
             
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
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / iters*num_instances, losses)
                a_real = model.real_A 
                b_fake = model.fake_B 
                save_img(a_real, b_fake, image_save_dir, epoch, total_iters)
                
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
