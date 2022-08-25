import torch
import torch.nn as nn
import numpy as np
from scipy.special import ndtri

class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='raw',iif_norm=0,reduction='mean',device='cuda',weight=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction='none',weight=weight)
        self.reduction=reduction
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
        
    def forward(self, pred, targets=None,infer=False):
        if infer is False:
            loss = self.loss_fcn(pred*self.iif[self.variant],targets)
            
            if self.reduction=='mean':
                loss=loss.mean()
            elif self.reduction=='sum':
                loss=loss.sum()
            return loss
        else:
            out = (pred*self.iif[self.variant])
            return out

        
class FocalLoss(nn.Module):
    def __init__(self,gamma,alpha=None,reduction='mean',device='cuda',weights=None):
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
    
    def set_weights(self,weights):
        self.weights=weights.unsqueeze(0)
        
    def forward(self, pred, targets):
        y_onehot = torch.cuda.FloatTensor(pred.shape)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        if self.gamma==0:
            loss =  self.loss_fn(pred,y_onehot)
            loss*=self.weights
            if self.reduction == 'sum':
                loss = loss.sum()/targets.shape[0]
            else:
                loss = loss.mean()
                
            return loss
        else:
            pred = torch.sigmoid(pred)
            loss = self.loss_fn(pred,y_onehot)
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

class Mixup(object):

    def __init__(self,criterion, alpha=1):
        self.alpha = alpha
        self.criterion=criterion

    def __call__(self, x, y, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)