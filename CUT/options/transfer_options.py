from .base_options import BaseOptions


class TransferOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--save_dir', type=str, default='/home/jun/ReID_Dataset/Market-1501-v15.09.15-test', help='resume training from another checkpoint')
        parser.add_argument('--source_dir', type=str, default='/home/jun/ReID_Dataset/Market-1501-v15.09.15', help='resume training from another checkpoint')
        self.isTrain = False
        return parser
