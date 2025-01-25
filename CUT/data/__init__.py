"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import random
from torch.utils.data import Sampler
from collections import defaultdict

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    if opt.dataset_mode == 'labeled':
        data_loader = LabeledDatasetDataLoader(opt) 
    else:
        data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class LabeledDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.batch_sampler = CustomBatchSampler(self.dataset, num_instances=2)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            batch_sampler=self.batch_sampler,
            num_workers=int(opt.num_threads),
        )

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            drop_last=True if opt.isTrain else False,
        )

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


class CustomBatchSampler(Sampler):
    def __init__(self, data_source, num_instances=2):
        self.data_source = data_source
        self.num_instances = num_instances
        self.aid_to_indices = defaultdict(list)
        self.bid_to_indices = defaultdict(list)

        for idx, data in enumerate(data_source):
            self.aid_to_indices[data['A_id']].append(idx)
            self.bid_to_indices[data['B_id']].append(idx)

        self.aids = list(self.aid_to_indices.keys())
        self.bids = list(self.bid_to_indices.keys())

    def __len__(self):
        return min(len(self.aids), len(self.bids)) * self.num_instances

    def __iter__(self):
        ret = []
        aid_indices = torch.randperm(len(self.aids)).tolist()
        bid_indices = torch.randperm(len(self.bids)).tolist()

        for aid_idx in aid_indices:
            aid = self.aids[aid_idx]
            a_indices = self.aid_to_indices[aid]
            if len(a_indices) < self.num_instances:
                a_indices = a_indices * (self.num_instances // len(a_indices)) + a_indices[:self.num_instances % len(a_indices)]
            a_indices = random.sample(a_indices, self.num_instances)
            ret.extend(a_indices)

        for bid_idx in bid_indices:
            bid = self.bids[bid_idx]
            b_indices = self.bid_to_indices[bid]
            if len(b_indices) < self.num_instances:
                b_indices = b_indices * (self.num_instances // len(b_indices)) + b_indices[:self.num_instances % len(b_indices)]
            b_indices = random.sample(b_indices, self.num_instances)
            ret.extend(b_indices)

        return iter(ret)

def No_index(items, item_to_exclude):
    return [index for index, item in enumerate(items) if item != item_to_exclude]


