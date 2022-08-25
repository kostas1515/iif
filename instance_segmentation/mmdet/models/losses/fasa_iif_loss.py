import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses import binary_cross_entropy, mask_cross_entropy
import pandas as pd
from .accuracy import accuracy
from .utils import weight_reduce_loss


@LOSSES.register_module()
class FasaIIFLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 use_cums=False,
                 num_classes=1203,
                 path='./lvis_files/idf_1204.csv',
                 variant='raw'):
        """FasaIIFLoss.
        """
        super(FasaIIFLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = self.cross_entropy

        self.num_classes = num_classes
        self.use_cums = use_cums
        if self.use_cums:
            self.open_cums()
        

        self.iif_weights =pd.read_csv(path)[variant].values.tolist()
            
        self.iif_weights = self.iif_weights[1:]+[1.0] #+1 for bg
        self.iif_weights = torch.tensor(self.iif_weights,device='cuda',dtype=torch.float).unsqueeze(0)
        
        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

    def open_cums(self):
        self.use_cums = True
        self.reduction_old = self.reduction
        self.reduction = 'none'
        self.cum_losses = torch.zeros(self.num_classes + 1).cuda()
        self.cum_labels = torch.zeros(self.num_classes + 1).cuda()

    def close_cums(self):
        self.use_cums = False
        self.reduction = self.reduction_old
        self.cum_losses = torch.zeros(self.num_classes + 1).cuda()
        self.cum_labels = torch.zeros(self.num_classes + 1).cuda()
    
    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """

        scores = torch.softmax(self.iif_weights*cls_score,dim=-1)

        return scores
    
    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 1
    
    def get_accuracy(self, cls_score, labels):
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.

        Returns:
            Dict [str, torch.Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        acc_classes = accuracy(cls_score, labels)
        acc = dict()
        acc['acc_classes'] = acc_classes
        
        return acc
    
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        if self.use_cums:
            unique_labels = label.unique()
            for u_l in unique_labels:
                inds_ = torch.where(label == u_l)[0]
                self.cum_labels[int(u_l)] += len(inds_)
                self.cum_losses[int(u_l)] += loss_cls[inds_].sum()
            loss_cls = loss_cls.mean()

        return loss_cls
    
    
    def cross_entropy(self,
                  pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100):
        
        """Calculate the CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the number
                of classes.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.
            ignore_index (int | None): The label index to be ignored.
                If None, it will be set to default value. Default: -100.

        Returns:
            torch.Tensor: The calculated loss
        """
        # The default value of ignore_index is the same as F.cross_entropy
        ignore_index = -100 if ignore_index is None else ignore_index
        # element-wise losses

        loss = F.cross_entropy(
            pred*self.iif_weights,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss
