from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC, DukeMTMC_marketstyle, DukeMTMC_all, DukeMTMC_cycleGAN, DukeMTMC_random, DukeMTMC_cut, DukeMTMC_marketstyle_spgan_segmentation,DukeMTMC_spgan_allcam, DukeMTMC_spgan_randomcam, DukeMTMC_spgan_random, DukeMTMC_stargan_randomcam, DukeMTMC_spcut_random, DukeMTMC_spcut_all, DukeMTMC_cut_random, DukeMTMC_cut_all, DukeMTMC_spgan_id_all, DukeMTMC_spgan_all, DukeMTMC_stargan_allcam
from .market1501 import Market1501, Market1501_dukestyle, Market1501_all, Market1501_cycleGAN, Market1501_random, Market1501_cut, Market1501_dukestyle_spgan_segmentation, Market1501_spgan_allcam, Market1501_spgan_randomcam, Market1501_spgan_random, Market1501_stargan_randomcam, Market1501_spcut_random, Market1501_spcut_all, Market1501_cut_random, Market1501_cut_all, Market1501_spgan_id_all, Market1501_spgan_all, Market1501_stargan_allcam
from .msmt17 import MSMT17
from .personx import personX
from .personxval import personXval
from .last import last
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'market1501_dukestyle': Market1501_dukestyle,
    'market1501_spgan_allcam': Market1501_spgan_allcam,
    'market1501_spgan_randomcam': Market1501_spgan_randomcam,
    
    'market1501_spgan_random': Market1501_spgan_random,
    'market1501_spgan_all': Market1501_spgan_all,
    
    'market1501_stargan_randomcam': Market1501_stargan_randomcam,
    'market1501_stargan_allcam': Market1501_stargan_allcam,
    
    'dukemtmc_marketstyle': DukeMTMC_marketstyle,
    'dukemtmc_spgan_allcam': DukeMTMC_spgan_allcam,
    'dukemtmc_spgan_randomcam': DukeMTMC_spgan_randomcam,
    
    'dukemtmc_spgan_random': DukeMTMC_spgan_random,
    'dukemtmc_spgan_all': DukeMTMC_spgan_all,
    
    'dukemtmc_stargan_randomcam': DukeMTMC_stargan_randomcam,
    'dukemtmc_stargan_allcam': DukeMTMC_stargan_allcam,
     
    'market1501_spcut_all': Market1501_spcut_all,
    'market1501_spcut_random': Market1501_spcut_random, 
    'market1501_cut_all': Market1501_cut_all,
    'market1501_cut_random': Market1501_cut_random,  
    'dukemtmc_spcut_all': DukeMTMC_spcut_all,
    'dukemtmc_spcut_random': DukeMTMC_spcut_random,
    'dukemtmc_cut_all': DukeMTMC_cut_all,
    'dukemtmc_cut_random': DukeMTMC_cut_random,
    
    'market1501_spgan_id_all': Market1501_spgan_id_all,
    'dukemtmc_spgan_id_all': DukeMTMC_spgan_id_all,
     
    'market1501_spgan_segmentation': Market1501_dukestyle_spgan_segmentation,
    'dukemtmc_spgan_segmentation': DukeMTMC_marketstyle_spgan_segmentation,
    'market1501_cycleGAN': Market1501_cycleGAN,
    'market1501_cut': Market1501_cut,
    'dukemtmc_cycleGAN': DukeMTMC_cycleGAN,
    'market1501_all': Market1501_all,
    'dukemtmc_all': DukeMTMC_all,
    'market1501_random': Market1501_random,
    'dukemtmc_random': DukeMTMC_random,
    'dukemtmc_cut': DukeMTMC_cut,
    'msmt17': MSMT17,
    'personx': personX,
    'personxval': personXval,
    'last': last
}


def names():
    return sorted(__factory.keys())


def create(name, root, l=1, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, ncl=l, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
