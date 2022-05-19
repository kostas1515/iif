from typing import OrderedDict
import torch
import torch.nn as nn
try:
    import imbalanced_dataset
    import presets
except ImportError:
    from classification import imbalanced_dataset
    from classification import presets

import os
import numpy as np
from scipy.special import ndtri,softmax
from sklearn.feature_selection import chi2,mutual_info_classif,f_classif
from sklearn.feature_selection import SelectKBest
import torch.distributed as dist

class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='raw',iif_norm=0,reduction='mean',device='cuda',weight=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction='none',weight=weight)
        self.reduction=reduction
        
#         self.loss_fcn = nn.MultiMarginLoss(reduction=reduction,weight=weight)
        if variant == 'log_adj':
            variant = 'raw'
            self.log_adj = True
        else:
            self.log_adj = False
            
        if (variant == 'hybrid')|(variant == 'hybrid_p'):
            self.beta = torch.nn.Parameter(0.0*torch.ones(1,device='cuda'))
            if (variant == 'hybrid_p'):
                self.gamma = torch.nn.Parameter(0.5*torch.ones(7,device='cuda'))
                self.hybrid_p = True
            else:
                self.hybrid_p = False
            variant = 'raw'
            self.hybrid = True
        else:
            self.hybrid = False
            self.hybrid_p = False
            
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
        
    def forward(self, pred, targets=None,infer=False):
        
#         weighted_prob=torch.abs(self.beta)
#         weighted_prob=torch.softmax(self.beta,dim=-1)
        if self.hybrid is True:
            weighted_prob=torch.sigmoid(self.beta)
            if self.hybrid_p is True:
                weighted_probs=torch.softmax(self.gamma,dim=-1)
    
        if infer is False:
            if self.log_adj is True:
                loss = self.loss_fcn(pred-self.iif[self.variant],targets)
            elif (self.hybrid is True)|(self.hybrid_p is True):
                if (self.hybrid_p is True):
                    multiplicative_loss=torch.stack([weighted_probs[counter]*self.loss_fcn(pred*v[1],targets) \
                                    for counter,v in enumerate(self.iif.items())],axis=0).sum(axis=0)
                    additive_loss=torch.stack([weighted_probs[counter]*self.loss_fcn(pred+v[1],targets) \
                                    for counter,v in enumerate(self.iif.items())],axis=0).sum(axis=0)
                    loss = weighted_prob*multiplicative_loss + (1-weighted_prob)*additive_loss
                else:
                    loss = weighted_prob*(self.loss_fcn(pred*self.iif[self.variant],targets)) + \
                           (1-weighted_prob)*(self.loss_fcn(pred+self.iif[self.variant],targets))
                
            else:
                loss = self.loss_fcn(pred*self.iif[self.variant],targets)
#                 loss = torch.stack([weighted_prob[counter]*self.loss_fcn(pred*v[1],targets) \
#                                     for counter,v in enumerate(self.iif.items())],axis=0).sum(axis=0)

            if self.reduction=='mean':
                loss=loss.mean()
            elif self.reduction=='sum':
                loss=loss.sum()
            return loss
        else:
            if (self.hybrid is True)|(self.hybrid_p is True):
                if (self.hybrid_p is True):
                    multiplicative_logits=torch.stack([weighted_probs[counter]*pred*v[1] \
                                    for counter,v in enumerate(self.iif.items())],axis=0).sum(axis=0)
                    additive_logits=torch.stack([weighted_probs[counter]*(pred+v[1]) \
                                    for counter,v in enumerate(self.iif.items())],axis=0).sum(axis=0)
                    out = weighted_prob*multiplicative_logits + (1-weighted_prob)*additive_logits
                else:
                    out = (weighted_prob)*(pred*self.iif[self.variant])+(1-weighted_prob)*(pred+self.iif[self.variant])
            else:
                out = (pred*self.iif[self.variant])

            return out


class TwoBranchLoss(nn.Module):
    def __init__(self,criterion1,criterion2,total_epochs):
        super(TwoBranchLoss,self).__init__()
        self.loss_fn1 = criterion1
        self.loss_fn2 = criterion2
    
    def forward(self, pred, targets=None,infer=False,epoch=0):
        pred1,pred2=pred
        if infer is False:
            loss1 = self.loss_fn1(pred1,targets)
            loss2 = self.loss_fn2(pred2,targets)
            loss=loss1+loss2

            return loss
        else:
            if hasattr(self.loss_fn1, 'iif'):
                pred1=self.loss_fn1(pred1,infer=True)
            if hasattr(self.loss_fn2, 'iif'):
                pred2=self.loss_fn2(pred2,infer=True)
            return (pred1+pred2)/2



class ADIIFLoss(nn.Module):
    def __init__(self,reduction='mean',device='cuda',weight=None):
        super(ADIIFLoss, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction='none',weight=weight)
        self.reduction=reduction
        self.iif = 'adaptive'
        
#         freqs = torch.tensor(list(dataset.num_per_cls_dict.values()),device='cuda',dtype=torch.float)
#         self.weights = torch.log(freqs.sum()/freqs).unsqueeze(0)

    def set_weights(self,weights):
        self.weights=weights

    def forward(self,pred,targets=None,infer=False):
            
        w = -torch.log(torch.softmax(pred,dim=-1))
        # w = torch.softmax(pred,dim=-1)
        batch_w=[torch.zeros_like(w) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_w,w)
        batch_w=torch.cat(batch_w,axis=0)
        avg_w = batch_w.mean(axis=0).unsqueeze(0)
        # print(avg_w)
        # avg_w = -torch.log(batch_w.mean(axis=0).unsqueeze(0))

        if infer is False:
            loss = self.loss_fcn(pred*avg_w,targets)

            if self.reduction=='mean':
                loss=loss.mean()
            elif self.reduction=='sum':
                loss=loss.sum()/pred.shape[0]
            return loss
        else:

            out = (pred*avg_w)

            return out



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
            # loss=loss.sum()

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

        candidates = torch.bincount(batch_targets,minlength=self.num_classes)
        candidates[candidates<2]=0
        candidates_indices = torch.nonzero(candidates)
        candidates=(candidates[candidates!=0])
        candidates=candidates.unsqueeze(1)
        new_images=[((10*torch.rand(cand,1,1,1) -5*torch.rand(1)).cuda()*batch_images[batch_targets==ind]).sum(axis=0) for ind,cand in zip(candidates_indices,candidates) for k in range(cand)]
        new_targets = [ind*torch.ones([cand],device='cuda',dtype=torch.long) for ind,cand in zip(candidates_indices,candidates)]
        new_images = torch.stack(new_images,axis=0)
        new_targets = torch.cat(new_targets,axis=0)
 
        # for i in range(self.num_classes):
        #     mask  = batch_targets==i
        #     images2blend  = batch_images[mask]
        #     group_size=images2blend.shape[0]
        #     if group_size>1:
        #         repeat = int(group_size*increase_factor)
        #         for j in range(repeat):
        #             # weights=torch.rand(group_size,1,1,1)
        #             weights=10*torch.rand(group_size,1,1,1) -5*torch.rand(1)
        #             weights=weights.cuda()
        #             new_images.append((images2blend*weights).sum(axis=0))
        #             new_targets.append(i)
        try:
            # new_targets=torch.tensor(new_targets,device='cuda',dtype=torch.long)
            # new_images = torch.stack(new_images,axis=0)
            final_images=torch.cat([new_images,image],axis=0)
            final_targets=torch.cat([new_targets,target],axis=0)
            return final_images,final_targets
        except RuntimeError:
            #new targets is empty
            return image,target


class ConLoss(nn.Module):
    def __init__(self,tau=0.1):
        super(ConLoss,self).__init__()
        self.tau = tau

    def forward(self, features, targets):
        sim_mat= (features).mm(features.T)
        norms= torch.norm(features,dim=1).unsqueeze(0)
        norm_mat = norms.T.mm(norms)
        norm_mat[norm_mat==0]=1e-8
        norm_sim_mat = sim_mat/norm_mat
        # norm_sim_mat=sim_mat

        mask=(targets.expand(targets.shape[0],targets.shape[0])==targets.unsqueeze(1))+0 #compute mask for positive examples
        exp_mat=torch.exp(norm_sim_mat/self.tau)
        upper_diag_sim_mat = torch.triu(exp_mat,1)
        lower_diag_sim_mat = torch.tril(exp_mat,-1)
        actual_sim=(upper_diag_sim_mat+lower_diag_sim_mat)
        # actual_sim=(upper_diag_sim_mat)/self.tau
        # normalised = (actual_sim/actual_sim.sum(axis=1).unsqueeze(1))
        normalised = actual_sim / (actual_sim+(actual_sim*((mask==0.0)+0)).sum(axis=1).unsqueeze(1))
        normalised = normalised +torch.eye(normalised.shape[0],normalised.shape[1],device='cuda',requires_grad=True)
        exp_mat=-torch.log(normalised)
        positive_cardinality= (mask.sum(axis=1).unsqueeze(1))
        exp_mat=exp_mat/positive_cardinality #devide by cardinality of positive samples rowise
        exp_mat[torch.isnan(exp_mat)]=0
        contrastive_loss = (exp_mat*mask).sum()/(positive_cardinality).sum()

        # if torch.isnan(contrastive_loss):
        #     contrastive_loss=0
        # contrastive_loss = (exp_mat*mask).sum()

        # contrastive_loss = (exp_mat*torch.triu(mask,1)).sum()

        # print(f'\r ration:{progress[mask==1].mean().item()/progress[mask==0].mean().item()},alpha:{self.alpha.item()}',end='', flush=True)
        # print(f'\r sim:{norm_sim_mat[mask==1].mean().item()},dissim:{norm_sim_mat[mask==0].mean().item()}',end='', flush=True)
        # loss =  self.loss_fn(logits,targets)
        # if self.reduction=='mean':
        #     loss=loss.mean()
        # elif self.reduction=='sum':
        #     loss=loss.sum()

        # print(self.alpha)
        # total_loss = (torch.sigmoid(self.alpha))*loss+(1-torch.sigmoid(self.alpha))*contrastive_loss
        # total_loss =contrastive_loss + loss
        # total_loss =contrastive_loss + 0.0*loss
        # print(self.alpha**3)

        return contrastive_loss


class ContrastiveLoss(nn.Module):
    def __init__(self,sup_loss,num_of_views,tau=0.1,total_epochs=200):
        super(ContrastiveLoss,self).__init__()
        self.loss_fn = sup_loss
        self.tau = tau
        self.num_of_views=num_of_views
        self.total_epochs=total_epochs
        self.alpha = 1
        # self.con_loss = SupConLoss()
        self.con_loss = ConLoss()


    def forward(self, pred,targets=None,infer=False,epoch=0):
        if infer is False:
            logits,features=pred
            # bs=targets.shape[0]
            # features=features.view(bs//self.num_of_views,self.num_of_views,-1)

            orig_mask=torch.arange(targets.shape[0])%self.num_of_views==0
            orig_targets = targets[orig_mask]

            # gpus=dist.get_world_size()
            # # batch_targets=[torch.zeros_like(orig_targets) for _ in range(gpus)]
            # batch_targets=[torch.zeros_like(targets) for _ in range(gpus)]
            # batch_features=[torch.zeros_like(features) for _ in range(gpus)]
            # # dist.all_gather(batch_targets,orig_targets)
            # dist.all_gather(batch_targets,targets)
            # dist.all_gather(batch_features,features)
            # batch_targets=torch.cat(batch_targets,axis=0)
            # batch_features=torch.cat(batch_features,axis=0)

            if self.num_of_views>1:
                # contrastive_loss=0.5*(self.con_loss(batch_features,batch_targets))+0.5*(self.con_loss(features,targets))
                contrastive_loss=self.con_loss(features,targets)
            else:
                contrastive_loss=0

            # loss =  self.loss_fn(logits,targets)
            loss =  self.loss_fn(logits[orig_mask],orig_targets)
            # loss=loss.mean()
            total_loss =contrastive_loss + loss
            self.alpha  =1-(epoch/self.total_epochs)**2
            # total_loss = (1-self.alpha)*loss+(self.alpha)*contrastive_loss

            return total_loss
        else:
            logits,features=pred
            if  hasattr(self.loss_fn, 'iif'):
                out = self.loss_fn(logits,infer=True)
            else:
                out=logits

            return out


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
