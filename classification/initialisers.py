import custom
import torch
import numpy as np
import presets
import imbalanced_dataset
import os
import time
import utils
import torchvision
import torchvision.transforms as transforms
from randaugment import CIFAR10Policy
import torchvision.datasets as datasets
from catalyst.data import  BalanceClassSampler,DistributedSamplerWrapper



def get_weights(dataset):
    per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
    per_cls_weights = per_cls_weights.sum()/per_cls_weights
    return per_cls_weights


def get_criterion(args,dataset,model,num_classes):
    if (args.classif== 'iif'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.IIFLoss(dataset,variant=args.iif,iif_norm=args.iif_norm,reduction=args.reduction,weight=weight)
    elif (args.classif== 'gombit'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.GombitLoss(len(dataset.classes),reduction=args.reduction,weights=weight)
        # torch.nn.init.constant_(model.linear.bias.data,np.log(np.log(1+1/(num_classes-1))))
        try:
            torch.nn.init.constant_(model.linear.bias.data,-np.log(np.log(num_classes)))
        except torch.nn.modules.module.ModuleAttributeError:
            print('Warning cannot set bias value for classifier')
            pass
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'bce'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.FocalLoss(gamma=0,reduction=args.reduction,feat_select=args.feat_select,weights=weight)
        torch.nn.init.constant_(model.linear.bias.data,-6.5)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'focal_loss'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.FocalLoss(gamma=args.gamma,alpha=args.alpha,reduction=args.reduction,feat_select=args.feat_select,weights=weight)
        torch.nn.init.constant_(model.linear.bias.data,-6.5)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'ce_loss'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.CELoss(feat_select=args.feat_select,weights=weight,reduction=args.reduction)
    elif (args.classif== 'gaussian'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.GaussianLoss(len(dataset.classes),reduction=args.reduction,weights=weight)
        torch.nn.init.constant_(model.linear.bias.data,-2)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'multiactivation'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.MultiActivationLoss(len(dataset.classes),reduction=args.reduction,weights=weight)
        torch.nn.init.constant_(model.linear.bias.data,-2)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'adiif'):
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = custom.ADIIFLoss(reduction=args.reduction,weight=weight)
    else:
        if args.deffered:
            weight=get_weights(dataset)
        else:
            weight=None
        criterion = torch.nn.CrossEntropyLoss(weight=weight,reduction=args.reduction)

    if args.contrastive_learning>0:
        criterion = custom.ContrastiveLoss(criterion,args.contrastive_learning,total_epochs=args.epochs)

    if args.tbl is True:
        criterion = custom.TwoBranchLoss(torch.nn.CrossEntropyLoss(),criterion,total_epochs=args.epochs)

    return criterion


def get_data(args):
    if args.dset_name =="ImageNet":
        num_classes = 1000
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_dir = os.path.join(args.data_path, 'train')
        val_dir = os.path.join(args.data_path, 'val')
        dataset = torchvision.datasets.ImageFolder(root=train_dir,transform=transform_train)
        dataset_test = torchvision.datasets.ImageFolder(root=val_dir,transform=transform_test)
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    elif args.dset_name =="imagenet_lt":
        num_classes = 1000
        train_txt = "../../../datasets/ImageNet-LT/ImageNet_LT_train.txt"
        eval_txt = "../../../datasets/ImageNet-LT/ImageNet_LT_test.txt"
        dataset, dataset_test, train_sampler, test_sampler = imbalanced_dataset.get_dataset_lt(args,num_classes,train_txt,eval_txt)
        num_classes = len(dataset.cls_num_list)
    elif args.dset_name =="inat18":
        num_classes = 8142
        # auto_augment_policy = getattr(args, "auto_augment", None)
        train_txt = "../../../datasets/train_val2018/iNaturalist18_train.txt"
        eval_txt = "../../../datasets/train_val2018/iNaturalist18_val.txt"
        dataset, dataset_test, train_sampler, test_sampler = imbalanced_dataset.get_dataset_lt(args,num_classes,train_txt,eval_txt)
        num_classes = len(dataset.cls_num_list)
    elif args.dset_name =="places_lt":
        num_classes = 365
        # auto_augment_policy = getattr(args, "auto_augment", None)
        train_txt = "../../../datasets/places365_standard/Places_LT_train.txt"
        eval_txt = "../../../datasets/places365_standard/Places_LT_test.txt"
        dataset, dataset_test, train_sampler, test_sampler = imbalanced_dataset.get_dataset_lt(args,num_classes,train_txt,eval_txt)
        num_classes = len(dataset.cls_num_list)
    else:
        dataset, dataset_test, train_sampler, test_sampler = load_cifar(args)
        num_classes = len(dataset.num_per_cls_dict)
    
    if args.contrastive_learning>0:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers, pin_memory=True,collate_fn=presets.my_collate)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    return dataset,num_classes,data_loader,data_loader_test,train_sampler



def load_cifar(args):
    
    if args.contrastive_learning>0:
        transform_train = transforms.Compose([presets.SimpleAugment(args.contrastive_learning)])
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        
        if auto_augment_policy=='cifar':
            transform_train=transforms.Compose(
                        [transforms.RandomCrop(32, padding=4), # fill parameter needs torchvision installed from source
                         transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
			             transforms.ToTensor(), 
                         presets.Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
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
        elif args.sampler=='weighted':
            class_weights = np.array(train_dataset.get_cls_num_list())

            class_weights = (class_weights - class_weights.min()) / (class_weights.max() - class_weights.min())
            class_weights = np.abs(class_weights-np.median(class_weights))
            class_weights = 10**(-class_weights)
            sample_weights = class_weights[train_dataset.targets]
            weighted_sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True)
            train_sampler= DistributedSamplerWrapper(weighted_sampler)
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

