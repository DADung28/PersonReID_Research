from __future__ import print_function, absolute_import
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import os.path as osp
import numpy as np

import glob
import re
import urllib
import zipfile
from torch.utils.data import  Dataset

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for item in data:
            pids += [item[1]]
            cams += [item[-1]]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_videodata_info(self, data, return_tracklet_stats=False):
        pids, cams, tracklet_stats = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_stats += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None

class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root, ncl=1, verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.ncl = ncl
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0

            if relabel: pid = pid2label[pid]
            pids=()
            for _ in range(self.ncl):
                pids=(pid,)+pids
            item=(img_path,) + pids + (camid,)
            dataset.append(item)
            # if relabel: pid = pid2label[pid]
            # dataset.append((img_path, pid, camid))

        return dataset


class DukeMTMC(BaseImageDataset):
    """
    DukeMTMC-reID
    Reference:
    1. Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
    2. Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.
    URL: https://github.com/layumi/DukeMTMC-reID_evaluation

    Dataset statistics:
    # identities: 1404 (train + query)
    # images:16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """
    dataset_dir = '.'

    def __init__(self, root, ncl=1, verbose=True, **kwargs):
        super(DukeMTMC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_test')
        self.camstyle_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_train_camstyle')

        self._download_data()
        self._check_before_run()

        self.ncl=ncl
        self.num_cam= 8
        # camstytrain = self._process_dir(self.camstylegallery_dir, relabel=True)
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading DukeMTMC-reID dataset")
        urllib.request.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            pids=()
            for _ in range(self.ncl):
                pids = (pid,)+pids
            item = (img_path,) + pids + (camid,)
            dataset.append(item)

        return dataset

class Market1501_singlecam(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root, ncl=1, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.ncl = ncl
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0

            if relabel: pid = pid2label[pid]
            pids=()
            for _ in range(self.ncl):
                pids=(pid,)+pids
            item=(img_path,) + pids + (0,)
            dataset.append(item)
            # if relabel: pid = pid2label[pid]
            # dataset.append((img_path, pid, camid))

        return dataset


class DukeMTMC_singlecam(BaseImageDataset):
    """
    DukeMTMC-reID
    Reference:
    1. Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
    2. Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.
    URL: https://github.com/layumi/DukeMTMC-reID_evaluation

    Dataset statistics:
    # identities: 1404 (train + query)
    # images:16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """
    dataset_dir = '.'

    def __init__(self, root, ncl=1, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_test')
        self.camstyle_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_train_camstyle')

        self._download_data()
        self._check_before_run()

        self.ncl=ncl
        self.num_cam= 8
        # camstytrain = self._process_dir(self.camstylegallery_dir, relabel=True)
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading DukeMTMC-reID dataset")
        urllib.request.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            pids=()
            for _ in range(self.ncl):
                pids = (pid,)+pids
            item = (img_path,) + pids + (0,)
            dataset.append(item)

        return dataset
 
class PersonReID(Dataset):
    def __init__(self, dataset1, dataset2, transform=None, mode = 'train', cam='multi'):
        super().__init__()
        self.mode = mode 
        if cam == 'multi':
            self.num_cams = dataset1.num_train_cams + dataset2.num_train_cams
        else: 
            dataset1.num_train_cams = 1
            self.num_cams = 2
        self.dataset = []#dataset
        for inds, item in enumerate(dataset1.train):
            self.dataset.append((item[0],item[1],item[2],inds))
        len_dataset1 = len(self.dataset)
        for inds, item in enumerate(dataset2.train):
            self.dataset.append((item[0],item[1],item[2]+dataset1.num_train_cams,inds+len_dataset1))
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        items = self.dataset[index] # fname, pid,pid1,pid2, camid, inds
        fname, camid, inds =items[0],items[-2],items[-1]
        pids = []
        for i, pid in enumerate(items[1:-2]):
            pids.append(pid)

        fpath = fname

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        camid_one_hot_vector = torch.zeros(self.num_cams)
        camid_one_hot_vector[camid] = 1 
        if self.mode == 'train':
            return img, camid_one_hot_vector
        else:
            return fname, img, camid_one_hot_vector


        
def get_re_id_loader(image_dir, crop_size=178, image_size=128, 
               batch_size=16, mode='train', cam='multi', num_workers=8):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.Resize((image_size,int(image_size/2)), Image.BICUBIC))
        transform.append(T.RandomCrop((crop_size, int(crop_size/2))))
    else:
        transform.append(T.Resize((crop_size,crop_size), Image.BICUBIC)) 
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if cam == 'multi':
        market = Market1501(image_dir)
        duke = DukeMTMC(image_dir)
    elif cam == 'single':
        market = Market1501_singlecam(image_dir)
        duke = DukeMTMC_singlecam(image_dir)
    
    dataset = PersonReID(market, duke, transform=transform, mode=mode, cam=cam)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
