from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys


from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from torch.nn import init
#
# from UDAsbs.utils.rerank import compute_jaccard_dist

from UDAsbs import datasets, sinkhornknopp as sk
from UDAsbs import models
from UDAsbs.trainers import DbscanBaseTrainer_unc_ema, DbscanBaseTrainer_unc_ema_single
from UDAsbs.evaluators import Evaluator, extract_features
from UDAsbs.utils.data import IterLoader
from UDAsbs.utils.data import transforms as T
from UDAsbs.utils.data.sampler import RandomMultipleGallerySampler
from UDAsbs.utils.data.preprocessor import Preprocessor
from UDAsbs.utils.logging import Logger
from UDAsbs.utils.serialization import load_checkpoint, save_checkpoint#, copy_state_dict

from UDAsbs.memorybank.NCEAverage import onlinememory
from UDAsbs.utils.faiss_rerank import compute_jaccard_distance
# import ipdb


start_epoch = best_mAP = 0

def get_data(name, data_dir, l=1):
    root = osp.join(data_dir)

    dataset = datasets.create(name, root, l)

    label_dict = {}
    for i, item_l in enumerate(dataset.train):
        # dataset.train[i]=(item_l[0],0,item_l[2])
        if item_l[1] in label_dict:
            label_dict[item_l[1]].append(i)
        else:
            label_dict[item_l[1]] = [i]

    return dataset, label_dict


def get_train_loader(dataset, height, width, choice_c, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.596, 0.558, 0.497])
    ])

    train_set = trainset #dataset.train if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances, choice_c)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    # train_loader = IterLoader(
    #     DataLoader(UnsupervisedCamStylePreprocessor(train_set, root=dataset.images_dir, transform=train_transformer,
    #                                                 num_cam=dataset.num_cam,camstyle_dir=dataset.camstyle_dir, mutual=True),
    #                batch_size=batch_size, num_workers=0, sampler=sampler,#workers
    #                shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

from UDAsbs.models.dsbn import convert_dsbn
from torch.nn import Parameter


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

def create_model(args, mb_h, ncs, wopre=False):
    model_1 = models.create(args.arch, mb_h = mb_h, num_features=args.features, dropout=args.dropout,
                            num_classes=ncs)

    model_1_ema = models.create(args.arch, mb_h = mb_h,  num_features=args.features, dropout=args.dropout,
                                num_classes=ncs)
    if not wopre:

        initial_weights = load_checkpoint(args.init_1)
        initial_weights_ema = load_checkpoint(args.init_2)
        copy_state_dict(initial_weights['state_dict'], model_1)
        copy_state_dict(initial_weights_ema['state_dict'], model_1_ema)
        print('load pretrain model:{}'.format(args.init_1))

    # adopt domain-specific BN
    convert_dsbn(model_1)
    convert_dsbn(model_1_ema)
    model_1.cuda()
    model_1_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_1_ema = nn.DataParallel(model_1_ema)

    for i, cl in enumerate(ncs):
        exec('model_1_ema.module.classifier{}_{}.weight.data.copy_(model_1.module.classifier{}_{}.weight.data)'.format(i,cl,i,cl))

    return model_1, None, model_1_ema, None

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


class Optimizer:
    def __init__(self, target_label, m, dis_gt, t_loader,N, hc=3, ncl=None,  n_epochs=200,
                 weight_decay=1e-5, ckpt_dir='/',fc_len=3500):
        self.num_epochs = n_epochs
        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.checkpoint_dir = ckpt_dir
        self.N=N
        self.resume = True
        self.checkpoint_dir = None
        self.writer = None
        # model stuff
        self.hc = len(ncl)#10
        self.K = ncl#3000
        self.K_c =[fc_len for _ in range(len(ncl))]
        self.model = m
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L = [torch.LongTensor(target_label[i]).to(self.dev) for i in range(len(self.K))]
        self.nmodel_gpus = 4#len()
        self.pseudo_loader = t_loader#torch.utils.data.DataLoader(t_loader,batch_size=256)
        # can also be DataLoader with less aug.
        self.train_loader = t_loader
        self.lamb = 25#args.lamb # the parameter lambda in the SK algorithm
        self.cpu=True
        self.dis_gt=dis_gt
        dtype_='f64'
        if dtype_ == 'f32':
            self.dtype = torch.float32 if not self.cpu else np.float32
        else:
            self.dtype = torch.float64 if not self.cpu else np.float64

        self.outs = self.K
        # activations of previous to last layer to be saved if using multiple heads.
        self.presize =  2048#4096 #

    def optimize_labels(self):
        if self.cpu:
            sk.cpu_sk(self)
        else:
            sk.gpu_sk(self)

        # save Label-assignments: optional
        # torch.save(self.L, os.path.join(self.checkpoint_dir, 'L', str(niter) + '_L.gz'))

        # free memory
        data = 0
        self.PS = 0

        return self.L

import collections


def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def write_sta_im(train_loader):
    label2num=collections.defaultdict(int)
    save_label=[]
    for x in train_loader:
        label2num[x[1]]+=1
        save_label.append(x[1])
    labels=sorted(label2num.items(),key=lambda item:item[1])[::-1]
    num = [j for i, j in labels]
    distribution = np.array(num)/len(train_loader)

    return num,save_label
def print_cluster_acc(label_dict,target_label_tmp):
    num_correct = 0
    for pid in label_dict:
        pid_index = np.asarray(label_dict[pid])
        pred_label = np.argmax(np.bincount(target_label_tmp[pid_index]))
        num_correct += (target_label_tmp[pid_index] == pred_label).astype(np.float32).sum()
    cluster_accuracy = num_correct / len(target_label_tmp)
    print(f'cluster accucary: {cluster_accuracy:.3f}')


class uncer(object):
    def __init__(self):
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        # self.cross_batch=CrossBatchMemory()
        self.kl_distance = nn.KLDivLoss(reduction='none')
    def kl_cal(self,pred1,pred1_ema):
        variance = torch.sum(self.kl_distance(self.log_sm(pred1),
                                              self.sm(pred1_ema.detach())), dim=1)
        exp_variance = torch.exp(-variance)
        return exp_variance

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters > 0) else None
    ncs = [int(x) for x in args.ncs.split(',')]
    # ncs_dbscan=ncs.copy()
    dataset_target, label_dict = get_data(args.dataset_target, args.data_dir, len(ncs))
    dataset_source, label_dict = get_data(args.dataset_source, args.data_dir, len(ncs))
    dataset_target_transfered, label_dict = get_data('market1501_dukestyle', args.data_dir, len(ncs))
    dataset_source_transfered, label_dict = get_data('dukemtmc_marketstyle', args.data_dir, len(ncs))
     
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    test_loader_source = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers)
    test_loader_target_transfered = get_test_loader(dataset_target_transfered, args.height, args.width, args.batch_size, args.workers) 
    test_loader_source_transfered = get_test_loader(dataset_source_transfered, args.height, args.width, args.batch_size, args.workers) 
    
    if args.arch in ['hrnet', 'ResNet50', 'resnet50_sbs', 'ResNet50_multi', 'resnet50_multi', 'resnet50_multi_sbs']: 
        mb_h = 2048
    elif args.arch == 'ViT':
        mb_h = 768
    elif args.arch == 'Swin':
        mb_h = 1024
    else:
        pass

    fc_len = 3500
    model_1, _, model_1_ema, _ = create_model(args, mb_h, [fc_len for _ in range(len(ncs))])
    # print(model_1)


    epoch = 0
    evaluator_1 = Evaluator(model_1)
    evaluator_1_ema = Evaluator(model_1_ema)


    (_, mAP_reranked_1), (_, mAP_1) = evaluator_1.evaluate(test_loader_target, dataset_target.query,
                                        dataset_target.gallery, cmc_flag=True, rerank=True)
    (_, mAP_reranked_1), (_, mAP_1) = evaluator_1.evaluate(test_loader_source, dataset_source.query,
                                        dataset_source.gallery, cmc_flag=True, rerank=True)
    (_, mAP_reranked_1), (_, mAP_1) = evaluator_1.evaluate(test_loader_target_transfered, dataset_target_transfered.query,
                                        dataset_target_transfered.gallery, cmc_flag=True, rerank=True)
    (_, mAP_reranked_1), (_, mAP_1) = evaluator_1.evaluate(test_loader_source_transfered, dataset_source_transfered.query,
                                        dataset_source_transfered.gallery, cmc_flag=True, rerank=True)
    
    mAP_2 = 0#evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
    #                                 cmc_flag=False)
    _, mAP_reranked_2 = 0,0#evaluator_1_ema.evaluate(test_loader_target, dataset_target.query,
    #                            dataset_target.gallery, cmc_flag=True, rerank=True)
    is_best = (mAP_1 > best_mAP) or (mAP_2 > best_mAP)
    best_mAP = max(mAP_1, mAP_2, best_mAP)

    print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%}, rerank: {:5.1%} model no.2 mAP: {:5.1%}, rerank: {:5.1%}  best: {:5.1%}{}\n'.
          format(epoch, mAP_1, mAP_reranked_1, mAP_2, mAP_reranked_2, best_mAP, ' *' if is_best else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT Training")
    # data
    parser.add_argument('-st', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-tt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--choice_c', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=700)
    parser.add_argument('--ncs', type=str, default='60')

    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    parser.add_argument('--height', type=int, default=224,
                        help="input height")
    parser.add_argument('--width', type=int, default=224,
                        help="input width")

    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='Swin',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer

    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--schedule_step', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--iters', type=int, default=200)

    parser.add_argument('--lambda-value', type=float, default=0)
    # training configs

    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    # parser.add_argument('--init-1', type=str, default='logs/personxTOpersonxval/resnet_ibn50a-pretrain-1_gem_RA//model_best.pth.tar', metavar='PATH')
    parser.add_argument('--init_1', type=str,
                        #default='/home/jun/UDAStrongBaseline/logs/market1501TOdukemtmc/Swin/model_best.pth.tar',
                        default='/home/jun/UDAStrongBaseline/logs/dukemtmcTOmarket1501/Swin/model_best.pth.tar',
                        #default='/home/jun/UDAStrongBaseline/logs/dbscan-market1501TOdukemtmc/Swin_uncertainly_new/model_best.pth.tar',
                        #default='logs/market1501TOdukemtmc/Swin/model_best.pth.tar',
                        metavar='PATH')
    parser.add_argument('--init_2', type=str,
                        default='/home/jun/UDAStrongBaseline/logs/dbscan-market1501TOdukemtmc/Swin_uncertainly/model2_checkpoint.pth.tar',
                        metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=16)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '/home/jun/ReID_Dataset'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default='logs/evalute_market1501TOdukemtmc/Swin_uncertainly')

    parser.add_argument('--lambda-tri', type=float, default=1.0)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--lambda-ct', type=float, default=0.05)
    parser.add_argument('--uncer-mode', type=float, default=0)#0 mean 1 max 2 min

    print("======mmt_train_dbscan_self-labeling=======")


    main()