import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class LabeledDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        # Create a mapping from original person IDs to new sequential IDs
        self.id_mapping_A = self.create_id_mapping(self.A_paths)
        self.id_mapping_B = self.create_id_mapping(self.B_paths, offset=len(self.id_mapping_A))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths, B_paths, A_id, and B_id
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            A_id (int)       -- sequential ID for domain A
            B_id (int)       -- sequential ID for domain B
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within the range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Extract person IDs from filenames and map to new sequential IDs
        original_A_id = self.extract_person_id(A_path)
        original_B_id = self.extract_person_id(B_path)
        A_id = self.id_mapping_A[original_A_id]
        B_id = self.id_mapping_B[original_B_id]

        # Apply image transformation
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_id': A_id, 'B_id': B_id}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def extract_person_id(self, path):
        """Extract person ID from the image filename.

        Parameters:
            path (str) -- image file path

        Returns:
            person_id (str) -- extracted person ID
        """
        filename = os.path.basename(path)
        person_id = filename.split('_')[0]
        return person_id

    def create_id_mapping(self, paths, offset=0):
        """Create a mapping from original person IDs to new sequential IDs.

        Parameters:
            paths (list) -- list of image paths
            offset (int) -- offset to apply to the new IDs (to ensure uniqueness)

        Returns:
            id_mapping (dict) -- dictionary mapping original person IDs to new sequential IDs
        """
        unique_ids = sorted(set(self.extract_person_id(path) for path in paths))
        id_mapping = {original_id: new_id + offset for new_id, original_id in enumerate(unique_ids)}
        return id_mapping
