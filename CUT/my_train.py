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

def display_and_save_batches_in_rows(batches, save_dir, epoch, i):
    """
    Display and save batches of images in rows.
    :param batches: List of 4 batches of images. Each batch is a tensor of shape (8, 3, 256, 256).
    :param save_dir: Directory to save the combined image.
    :param epoch: Current epoch number.
    :param i: Image index.
    """
    assert len(batches) == 4, "There should be exactly 4 batches"
    
    # Convert all images in batches to PIL images
    pil_images = [[tensor_to_pil(img) for img in batch] for batch in batches]

    # Calculate the size of the final image
    img_width, img_height = pil_images[0][0].size
    num_batches = len(batches)
    images_per_batch = len(batches[0])
    
    combined_image = Image.new('RGB', (img_width * images_per_batch, img_height * num_batches))

    # Paste images into the combined image
    for row_idx, batch in enumerate(pil_images):
        for col_idx, img in enumerate(batch):
            combined_image.paste(img, (col_idx * img_width, row_idx * img_height))
    
    # Save the combined image
    file_name = f'Epoch_{epoch}_batch_{i + 1}.jpg'
    file_path = os.path.join(save_dir, file_name)
    combined_image.save(file_path)



def get_data(data_dir, load_size, crop_size, batch_size=4, workers=16, num_instances=2, iters = 3000):
    A = 'market1501'
    B = 'dukemtmc'
    root = osp.join(data_dir)
    dataset_A = datasets.create(A, root)
    train_A = dataset_A.train
    dataset_B = datasets.create(B, root)
    train_B = dataset_B.train

    train_transformer = T.Compose([
            T.Resize((load_size,int(load_size/2)), Image.BICUBIC),
            T.RandomCrop((crop_size, int(crop_size/2))),
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
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    B_loader = IterLoader(DataLoader(Preprocessor(train_B, root=dataset_B.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler_B,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return A_loader, B_loader

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    image_save_dir = opt.image_save_dir 
    os.makedirs(image_save_dir, exist_ok=True)
    model = create_model(opt)      # create a model given opt.model and other options
    #print('The number of training images = %d' % dataset_size)
    num_instances = 2
    iters = opt.iters
    datasetA, datasetB = get_data(opt.dataroot, opt.load_size, opt.crop_size, batch_size=opt.batch_size, workers=16, num_instances=num_instances, iters = opt.iters)
    iters = min(len(datasetA),len(datasetB))
    
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
            #print(data['A'].shape)
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
                #print(model.get_current_visuals()[0].shape)
                image = list(model.get_current_visuals().values())
                display_and_save_batches_in_rows(image, image_save_dir, epoch, i)
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, iters*batch_size, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / iters*num_instances, losses)

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
