from typing import OrderedDict
import torch
import torch.nn as nn
import imbalanced_dataset
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import os
import numpy as np
from scipy.special import ndtri
from sklearn.feature_selection import chi2,mutual_info_classif,f_classif
from sklearn.feature_selection import SelectKBest
import torch.distributed as dist

class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='raw',iif_norm=0,reduction='mean',device='cuda',weight=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction=reduction,weight=weight)
#         self.loss_fcn = nn.MultiMarginLoss(reduction=reduction,weight=weight)
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
        if iif_norm >0:
            self.iif = {k: v/torch.norm(v,p=iif_norm)  for k, v in self.iif.items()}
        print(self.iif[self.variant])
        
    def forward(self, pred, targets):
        loss = self.loss_fcn(pred*self.iif[self.variant],targets)
        return loss
    
class GombitLoss(nn.Module):
    def __init__(self,dataset,reduction='mean',device='cuda'):
        super(GombitLoss,self).__init__()
        self.loss_fn = nn.BCELoss(reduction='none').to(device)
        self.reduction=reduction
#         freqs = torch.tensor(list(dataset.num_per_cls_dict.values()),device='cuda',dtype=torch.float)
#         self.weights = torch.log(freqs.sum()/freqs).unsqueeze(0)
        
    def forward(self, pred, targets):
        y_onehot = torch.cuda.FloatTensor(pred.shape)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        pred=torch.clamp(pred,min=-10,max=2)
        pestim=1- 1/(torch.exp(torch.exp((pred))))
        loss =  self.loss_fn(pestim,y_onehot)
#         loss*=self.weights
        
#         neg_grad = (torch.exp(pred)*(1-y_onehot)).sum()
#         pos_grad = (torch.exp(pred)/(torch.exp(torch.exp(pred))-1))*(y_onehot)
#         print(f'pos grad is:{pos_grad.sum()}, neg grad is:{neg_grad.sum()}')
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/targets.shape[0]
        
        return loss
    
class CELoss(nn.Module):
    def __init__(self,feat_select=None,reduction='mean',device='cuda',weights=None):
        super(CELoss,self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none',weight=weights)
        self.reduction=reduction
        self.feat_select=feat_select
        
    def forward(self, pred, targets):    
        if (self.feat_select):
            with torch.no_grad():
                pestim= [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
                y=[torch.zeros_like(targets) for _ in range(dist.get_world_size())]
                dist.all_gather(y, targets)
                y=torch.cat(y)
                dist.all_gather(pestim, pred) 
                pestim=torch.cat(pestim)             
                if self.feat_select!='f_classif':
                    pestim=torch.softmax(pestim,dim=1)
                if self.feat_select=='entropy':
                    selector_scores = -(pestim*torch.log(pestim)/np.log(pestim.shape[1])).sum(axis=0)
                    selector_scores=selector_scores.unsqueeze(0)
                else:
                    X= pestim.cpu().detach().numpy()
                    y = y.cpu().detach().numpy()
                    selector = SelectKBest(eval(self.feat_select), k='all')
                    _ = selector.fit(X, y)
                    selector_scores = torch.tensor(selector.scores_,dtype=torch.float,device='cuda').unsqueeze(0)
#                 weights /=torch.norm(weights,2) 
        else:
            selector_scores = 1
        
        loss =  self.loss_fn(selector_scores*pred,targets)
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/targets.shape[0]
        
        return loss
        
class FocalLoss(nn.Module):
    def __init__(self,gamma,alpha=None,reduction='mean',device='cuda',feat_select=None,weights=None):
        super(FocalLoss,self).__init__()
        if gamma == 0:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
        else:
            self.loss_fn = nn.BCELoss(reduction='none').to(device)
        if (weights is not None):
            self.weights=weights.unsqueeze(0)
        else:
            self.weights=1
        self.gamma=gamma
        self.alpha = alpha
        self.reduction = reduction
        self.feat_select= feat_select
        
    def forward(self, pred, targets):
        y_onehot = torch.cuda.FloatTensor(pred.shape)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        # use mi
        if (self.feat_select):
            try:
                if self.feat_select!='f_classif':
                    with torch.no_grad():
                        pestim=torch.sigmoid(pred)
                else:
                    pestim = pred.clone().detach()
                X= pestim.cpu().detach().numpy()
                y = targets.cpu().detach().numpy()
                selector = SelectKBest(eval(self.feat_select), k='all')
                _ = selector.fit(X, y)
                selector_scores = torch.tensor(selector.scores_,dtype=torch.float,device='cuda').unsqueeze(0)
            except:
                selector_scores = 1
                pass   
        else:
            selector_scores = 1
        #end use ch
        if self.gamma==0:
            loss =  self.loss_fn(selector_scores*pred,y_onehot)
            loss*=self.weights
            if self.reduction == 'sum':
                loss = loss.sum()/targets.shape[0]
            else:
                loss = loss.mean()
                
            return loss
        else:
            pred = torch.sigmoid(pred)
            loss = self.loss_fn(selector_scores*pred,y_onehot)
            p_t = pred * y_onehot + (1 - pred) * (1 - y_onehot)
            loss = loss * ((1 - p_t) ** self.gamma)
            loss*=self.weights
            if (self.alpha):
                alpha_t = self.alpha * y_onehot + (1 - self.alpha) * (1 - y_onehot)
                loss = alpha_t * loss
            if self.reduction == 'sum':
                loss=loss.sum()
                loss = loss/targets.shape[0]
            else:
                loss=loss.mean()
            
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
        train_dataset = imbalanced_dataset.IMBALANCECIFAR10(root='../../../datasets/',
                                                            imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                            rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='../../../datasets/', train=False, download=True, transform=transform_val)
    elif args.dset_name == 'cifar100':
        train_dataset = imbalanced_dataset.IMBALANCECIFAR100(root='../../../datasets/',
                                                             imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                             rand_number=args.rand_number, train=True, download=True, transform=transform_train)
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


