import custom
import torch
import numpy as np
import presets
import imbalanced_dataset
import os
import time
import utils
import torchvision


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
        torch.nn.init.constant_(model.linear.bias.data,-np.log(np.log(num_classes)))
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
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.contrastive_learning>0:
        criterion = custom.ContrastiveLoss(criterion,args.contrastive_learning,total_epochs=args.epochs)

    if args.tbl is True:
        criterion = custom.TwoBranchLoss(torch.nn.CrossEntropyLoss(),criterion,total_epochs=args.epochs)

    

    return criterion

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision",
                              "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    resize_size, crop_size = (
        342, 299) if args.model == 'inception_v3' else (256, 224)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = imbalanced_dataset.IMBALANCEImageNet(
            traindir,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            rand_number=args.rand_number,
            transform = presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                              random_erase_prob=random_erase_prob))
        print(dataset)
        if args.cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))
        if args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def get_data(args):

    if args.dset_name =="ImageNet":
        train_dir = os.path.join(args.data_path, 'train')
        val_dir = os.path.join(args.data_path, 'val')
        dataset, dataset_test, train_sampler, test_sampler = load_data(
            train_dir, val_dir, args)
        num_classes = len(dataset.classes)
    elif args.dset_name =="imagenet_lt":
        auto_augment_policy = getattr(args, "auto_augment", None)
        dataset, dataset_test, train_sampler, test_sampler = imbalanced_dataset.get_imagenet_lt(args.distributed, root=args.data_path,
                              auto_augment=auto_augment_policy,sampler = args.sampler)
        num_classes = len(dataset.cls_num_list)
    else:
        dataset, dataset_test, train_sampler, test_sampler = custom.load_cifar(args)
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