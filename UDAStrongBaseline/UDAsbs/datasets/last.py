from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class last(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'last'

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
            print("=> LaST loaded")
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
        #pattern = re.compile(r'(\d+)_{2,}(\d+)_(\d+)_(\d+)_(\d+)_(\d)')
        pattern = re.compile(r'(\d+)__*(\d+)_(\d+)_(\d+)_(\d+)_(\d+)')
        pid_container = set()
        for img_path in img_paths:
            #print(img_path)
            pid, _, _, _, _, clothes_id = map(int, pattern.search(img_path).groups())
            #if pid == 0 or clothes_id == 0: continue  # junk images or unlabeled clothes are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, img_id, vid_id, frame_id, bbox_id, clothes_id = map(int, pattern.search(img_path).groups())
            #if pid == 0 or clothes_id == 0: continue  # junk images or unlabeled clothes are just ignored
            if relabel: pid = pid2label[pid]
            pids = ()
            for _ in range(self.ncl):
                pids = (pid,) + pids
            item = (img_path,) + pids + (clothes_id,)
            dataset.append(item)

        return dataset

    
