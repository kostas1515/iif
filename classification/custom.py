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
from catalyst.data import  BalanceClassSampler,DistributedSamplerWrapper
import math

class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='raw',iif_norm=0,reduction='mean',device='cuda',weight=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction='none',weight=weight)
        self.reduction=reduction
#         self.loss_fcn = nn.MultiMarginLoss(reduction=reduction,weight=weight)
        self.variant = variant
        freqs = np.array(dataset.get_cls_num_list())
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
#         print(self.iif[self.variant])
        
    def forward(self, pred, targets):
        loss = self.loss_fcn(pred*self.iif[self.variant],targets)
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()
        return loss
    
class GombitLoss(nn.Module):
    def __init__(self,num_classes,reduction='mean',device='cuda',weights=None):
        super(GombitLoss,self).__init__()
        self.loss_fn = nn.BCELoss(reduction='none').to(device)
        self.reduction=reduction
        if (weights is not None): 
            self.weights=weights
        else:
            self.weights=torch.ones(num_classes,device='cuda')
        
#         freqs = torch.tensor(list(dataset.num_per_cls_dict.values()),device='cuda',dtype=torch.float)
#         self.weights = torch.log(freqs.sum()/freqs).unsqueeze(0)

    def set_weights(self,weights):
        self.weights=weights
        
    def forward(self, pred, targets):
        y_onehot = torch.cuda.FloatTensor(pred.shape)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        pred=torch.clamp(pred,min=-4,max=10)
        pestim=1/(torch.exp(torch.exp((-pred))))
        loss =  self.loss_fn(pestim,y_onehot)
        loss*=self.weights[targets].unsqueeze(1)
        
#         pos_grad = (torch.exp(-pred)*(y_onehot)).sum()
#         neg_grad = (torch.exp(-pred)/(torch.exp(torch.exp(-pred))-1))*(1-y_onehot)
#         print(f'pos grad is:{pos_grad.sum()}, neg grad is:{neg_grad.sum()}')
        
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/targets.shape[0]
        
        return loss
    
# class GammaLoss(nn.Module):
#     def __init__(self,num_classes,reduction='mean',device='cuda',weights=None):
#         super(GammaLoss,self).__init__()
#         self.loss_fn = nn.BCELoss(reduction='none').to(device)
#         self.reduction=reduction
#         if (weights is not None): 
#             self.weights=weights
#         else:
#             self.weights=torch.ones(num_classes,device='cuda')
        
# #         freqs = torch.tensor(list(dataset.num_per_cls_dict.values()),device='cuda',dtype=torch.float)
# #         self.weights = torch.log(freqs.sum()/freqs).unsqueeze(0)

#     def set_weights(self,weights):
#         self.weights=weights
        
#     def forward(self, pred, targets):
#         y_onehot = torch.cuda.FloatTensor(pred.shape)
#         y_onehot.zero_()
#         y_onehot.scatter_(1, targets.unsqueeze(1), 1)
# #         pred=torch.clamp(pred,min=-4,max=10)
#         pestim=torch.igamma(torch.exp(pred),1)
#         loss =  self.loss_fn(pestim,y_onehot)
#         loss*=self.weights[targets].unsqueeze(1)
        
# #         pos_grad = (torch.exp(-pred)*(y_onehot)).sum()
# #         neg_grad = (torch.exp(-pred)/(torch.exp(torch.exp(-pred))-1))*(1-y_onehot)
# #         print(f'pos grad is:{pos_grad.sum()}, neg grad is:{neg_grad.sum()}')
        
#         if self.reduction=='mean':
#             loss=loss.mean()
#         elif self.reduction=='sum':
#             loss=loss.sum()/targets.shape[0]
        
#         return loss
    
class GaussianLoss(nn.Module):
    def __init__(self,num_classes,reduction='mean',device='cuda',weights=None):
        super(GaussianLoss,self).__init__()
        self.loss_fn = nn.BCELoss(reduction='none').to(device)
        self.reduction=reduction
        if (weights is not None): 
            self.weights=weights
        else:
            self.weights=torch.ones(num_classes,device='cuda')
#         freqs = torch.tensor(list(dataset.num_per_cls_dict.values()),device='cuda',dtype=torch.float)
#         self.weights = torch.log(freqs.sum()/freqs).unsqueeze(0)
    def set_weights(self,weights):
        self.weights=weights
        
    def gaussian_cdf(self,input):
        mu=0
        sigma=1
        x=(input-mu)/(2**(1/2)*sigma)
        a=0.147
        res= torch.sign(x)*(1-torch.exp(-(x**2)*((4/np.pi+a*(x**2))/(1+a*(x**2)))))**(1/2)
        return (1+res)/2
        
    def forward(self, pred, targets):
        y_onehot = torch.cuda.FloatTensor(pred.shape)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        pred=torch.clamp(pred,min=-8,max=8)
        pestim=1/2+torch.erf(pred/(2**(1/2)))/2
        loss =  self.loss_fn(pestim,y_onehot)
        loss=torch.clamp(loss,min=0,max=20)
        loss*=self.weights[targets].unsqueeze(1)
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/targets.shape[0]
        
        return loss

class GaussianLoss(nn.Module):
    def __init__(self,dataset,reduction='mean',device='cuda'):
        super(GaussianLoss,self).__init__()
        self.loss_fn = nn.BCELoss(reduction='none').to(device)
        self.reduction=reduction
#         freqs = torch.tensor(list(dataset.num_per_cls_dict.values()),device='cuda',dtype=torch.float)
#         self.weights = torch.log(freqs.sum()/freqs).unsqueeze(0)

    def gaussian_cdf(self,input):
        x=input/(2**(1/2))
        a=0.147
        res= torch.sign(x)*(1-torch.exp(-(x**2)*((4/math.pi+a*(x**2))/(1+a*(x**2)))))**(1/2)
        return (1+res)/2
        
    def forward(self, pred, targets):
        y_onehot = torch.cuda.FloatTensor(pred.shape)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        pestim=self.gaussian_cdf(pred)
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
    
class MultiActivationLoss(nn.Module):
    def __init__(self,num_classes,reduction='mean',device='cuda',weights=None):
        super(MultiActivationLoss,self).__init__()
        self.loss_fn = nn.BCELoss(reduction='none').to(device)
        self.reduction=reduction
        self.beta = torch.nn.Parameter(torch.ones(3,device='cuda'))
        if (weights is not None): 
            self.weights=weights
        else:
            self.weights=torch.ones(num_classes,device='cuda')

    def set_weights(self,weights):
        self.weights=weights
        
    def forward(self, pred, targets):
        y_onehot = torch.cuda.FloatTensor(pred.shape)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        gumbel_p=1/(torch.exp(torch.exp((-torch.clamp(pred,min=-4,max=10)))))
        normal_p=1/2+torch.erf(torch.clamp(pred,min=-8,max=8)/(2**(1/2)))/2
        logistic_p=torch.sigmoid(pred)
        weighted_prob=torch.softmax(self.beta,dim=0)     
#         print(f'gumbel:{weighted_prob[0]},normal:{weighted_prob[1]},logistic:{weighted_prob[2]}')
        pestim=gumbel_p*weighted_prob[0]+normal_p*weighted_prob[1]+logistic_p*weighted_prob[2]
        
        loss =  self.loss_fn(pestim,y_onehot)
        loss*=self.weights[targets].unsqueeze(1)
        
#         pos_grad = (torch.exp(-pred)*(y_onehot)).sum()
#         neg_grad = (torch.exp(-pred)/(torch.exp(torch.exp(-pred))-1))*(1-y_onehot)
#         print(f'pos grad is:{pos_grad.sum()}, neg grad is:{neg_grad.sum()}')
        
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
    
    def set_weights(self,weights):
        self.weights=weights.unsqueeze(0)
        
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
        if args.sampler=='random':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            train_labels = train_dataset.targets
            balanced_sampler = BalanceClassSampler(train_labels,mode=args.sampler)
            train_sampler= DistributedSamplerWrapper(balanced_sampler)

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        if args.sampler=='random':
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
        else:
            train_labels = train_dataset.targets
            train_sampler = BalanceClassSampler(train_labels,mode=args.sampler)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    return train_dataset, val_dataset, train_sampler, test_sampler

class LinearCombine(object):
    """Mix images

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self,num_classes,increase_factor=1):
        self.increase_factor=increase_factor
        self.num_classes=num_classes
    

    def __call__(self, image,target):
        new_images=[]
        new_targets=[]
        increase_factor=self.increase_factor
        batch_targets=[torch.zeros_like(target) for _ in range(dist.get_world_size())]
        batch_images=[torch.zeros_like(image) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_targets,target)
        dist.all_gather(batch_images,image)
        batch_targets=torch.cat(batch_targets,axis=0)
        batch_images=torch.cat(batch_images,axis=0)
        for i in range(self.num_classes):
            mask  = batch_targets==i
            if (mask.sum()>2):
                indices = torch.nonzero(mask)
                perm = torch.randperm(indices.size(0))
                idx = perm[:2]
                samples = indices[idx].squeeze(1)

                images2blend  = batch_images[samples]
                group_size=images2blend.shape[0]
                if group_size>1:
                    repeat = int(group_size*increase_factor)
                    for j in range(repeat):
#                         weights=10*torch.rand(group_size,1,1,1) -5*torch.rand(1)
                        weights=torch.rand(group_size,1,1,1)
                        weights=weights.cuda()
                        new_images.append((images2blend*weights).sum(axis=0))
                        new_targets.append(i)
        new_targets=torch.tensor(new_targets,device='cuda',dtype=torch.long)
        new_images = torch.stack(new_images,axis=0)
        final_images=torch.cat([new_images,image],axis=0)
        final_targets=torch.cat([new_targets,target],axis=0)
        return final_images,final_targets


