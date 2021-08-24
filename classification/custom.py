from typing import OrderedDict
import torch
import torch.nn as nn
import imbalanced_dataset
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import os
import numpy as np
from scipy.special import ndtri

class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='raw',precomputed=True,reduction='mean',device='cuda',weight=None):
        super(IIFLoss, self).__init__()
#         self.loss_fcn = nn.CrossEntropyLoss(reduction=reduction,weight=weight)
        self.loss_fcn = nn.MultiMarginLoss(reduction=reduction,weight=weight)
        self.precomputed = precomputed
        self.variant = variant
        freqs = np.array(list(dataset.num_per_cls_dict.values()))
        iif={}
        iif['raw']= np.log(freqs.sum()/freqs)
        iif['smooth'] = np.log((freqs.sum()+1)/(freqs+1))+1
        iif['rel'] = np.log((freqs.sum()-freqs)/freqs)
        
        iif['normit'] = -ndtri(freqs/freqs.sum())
        iif['gombit'] = -np.log(-np.log(1-(freqs/freqs.sum())))
        iif['base2'] = np.log2(freqs.sum()/freqs)
        iif['base10'] = np.log10(freqs.sum()/freqs)
        self.iif = {k: torch.tensor([v],dtype=torch.float).to(device,non_blocking=True) for k, v in iif.items()}
        
    def forward(self, pred, targets):
        if self.precomputed is True:
            loss = self.loss_fcn(pred*self.iif[self.variant],targets)
        return loss
    
    
def load_cifar(args):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dset_name == 'cifar10':
        train_dataset = imbalanced_dataset.IMBALANCECIFAR10(root='../../../datasets/', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='../../../datasets/', train=False, download=True, transform=transform_val)
    elif args.dset_name == 'cifar100':
        train_dataset = imbalanced_dataset.IMBALANCECIFAR100(root='../../../datasets/', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='../../../datasets/', train=False, download=True, transform=transform_val)
    

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    return train_dataset, val_dataset, train_sampler, test_sampler


