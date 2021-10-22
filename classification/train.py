import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import imbalanced_dataset

import presets
import utils
import custom
import resnet_cifar
import numpy as np
from sklearn.feature_selection import chi2,mutual_info_classif,f_classif
from sklearn.feature_selection import SelectKBest
from custom import LinearCombine

try:
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args):
    model.train()
    print_freq=args.print_freq
    apex=args.apex
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    
    lr_scheduler = None
    if epoch <5:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, 5*len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        
    if epoch ==5:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    lincomb=LinearCombine(len(data_loader.dataset.classes),increase_factor=args.lincomb_if)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        if args.schedule=='lincomb':
            image, target=lincomb(image, target)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size /
                                             (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
#             loss = criterion(output, target)
            weights = 1
            if  hasattr(criterion, 'iif'):
                weights = criterion.iif[criterion.variant]                
            acc1, acc5 = utils.accuracy(output*weights , target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
#             metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


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


def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    
    if args.dset_name =="ImageNet":
        train_dir = os.path.join(args.data_path, 'train')
        val_dir = os.path.join(args.data_path, 'val')
        dataset, dataset_test, train_sampler, test_sampler = load_data(
            train_dir, val_dir, args)
        num_classes = len(dataset.classes)
    else:
        dataset, dataset_test, train_sampler, test_sampler = custom.load_cifar(args)
        num_classes = len(dataset.num_per_cls_dict)
        
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    try:
        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained,num_classes=num_classes)
    except KeyError:
        #model does not exist in pytorch load it from resnset_cifar
        model = eval(f'resnet_cifar.{args.model}(num_classes={num_classes})')
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if (args.classif== 'iif'):
        per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
        per_cls_weights = per_cls_weights.sum()/per_cls_weights
        per_cls_weights /=per_cls_weights.sum()
        criterion = custom.IIFLoss(dataset,variant=args.iif,iif_norm=args.iif_norm,reduction=args.reduction,weight=per_cls_weights)
    elif (args.classif== 'gombit'):
#         per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
#         per_cls_weights = per_cls_weights.sum()/per_cls_weights
        criterion = custom.GombitLoss(len(dataset.classes),reduction=args.reduction)
#         torch.nn.init.constant_(model.linear.bias.data,np.log(np.log(1+1/(num_classes-1))))
        torch.nn.init.constant_(model.linear.bias.data,-np.log(np.log(num_classes)))
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'bce'):
#         per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
#         per_cls_weights = per_cls_weights.sum()/per_cls_weights
#         per_cls_weights /=per_cls_weights.sum()
        criterion = custom.FocalLoss(gamma=0,reduction=args.reduction,feat_select=args.feat_select)
        torch.nn.init.constant_(model.linear.bias.data,-6.5)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'focal_loss'):
#         per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
#         per_cls_weights = per_cls_weights.sum()/per_cls_weights
#         per_cls_weights /=per_cls_weights.sum()
        criterion = custom.FocalLoss(gamma=args.gamma,alpha=args.alpha,reduction=args.reduction,feat_select=args.feat_select)
        torch.nn.init.constant_(model.linear.bias.data,-6.5)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'ce_loss'):
        per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
        per_cls_weights = per_cls_weights.sum()/per_cls_weights
#         per_cls_weights = per_cls_weights/torch.norm(per_cls_weights,p=2)
        criterion = custom.CELoss(feat_select=args.feat_select,weights=per_cls_weights)
    elif (args.classif== 'gaussian'):
#         per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
#         per_cls_weights = per_cls_weights.sum()/per_cls_weights
        criterion = custom.GaussianLoss(len(dataset.classes),reduction=args.reduction)
        torch.nn.init.constant_(model.linear.bias.data,-2)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    elif (args.classif== 'multiactivation'):
        criterion = custom.MultiActivationLoss(len(dataset.classes),reduction=args.reduction)
        torch.nn.init.constant_(model.linear.bias.data,-2)
        torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
    else:
        criterion = nn.CrossEntropyLoss()

    opt_name = args.opt.lower()
    if (args.classif== 'multiactivation'):
        model_parameters=list(model.parameters())+list(criterion.parameters())
    else: 
        model_parameters=model.parameters()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model_parameters, lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    else:
        raise RuntimeError(
            "Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )
        
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return
    
    cls_num_list = dataset.get_cls_num_list()
    print("Start training")
    start_time = time.time()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if (epoch>159):
            if args.schedule=='lincomb':
                args.schedule='cost'
        if epoch>179:
            if args.schedule=='cost':
                per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
                per_cls_weights = per_cls_weights.sum()/per_cls_weights
                if args.classif=='ce':
                    criterion = custom.CELoss(feat_select=args.feat_select,weights=per_cls_weights) 
                else:
                    criterion.set_weights(per_cls_weights)
                    
            elif args.schedule=='iif':
#                 per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
#                 per_cls_weights = per_cls_weights.sum()/per_cls_weights
                criterion = custom.IIFLoss(dataset,variant=args.iif,iif_norm=args.iif_norm,reduction=args.reduction)
                
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader,
                        device, epoch, args)
        lr_scheduler.step()
        acc = evaluate(model, criterion, data_loader_test, device=device)
        if acc>best_acc:
            best_acc = acc
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('best acc is:',best_acc)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument(
        '--data-path', default='../../../datasets/ILSVRC/Data/CLS-LOC/', help='dataset')
    parser.add_argument(
        '--dset_name', default='ImageNet', help='dataset name')
    parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--milestones',nargs='+', default=[160,180],type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100,
                        type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--classif', default='ce',type=str, help='Type of classification')
    parser.add_argument('--gamma', default=2.0,type=float, help='Focal loss gamma hp')
    parser.add_argument('--alpha', default=None,type=float, help='Focal loss alpha hp')
    parser.add_argument('--iif', default='raw',type=str, help='Type of IIF variant')
    parser.add_argument('--iif_norm', default=0, type=int, help='IIF norm')
    parser.add_argument('--feat_select', default=None, type=str, help='pick either chi2 or mutual_info_classif')
    parser.add_argument('--schedule', default='normal', type=str, help='strategy of loss functions')
    parser.add_argument('--lincomb_if', default=1.0, type=float, help='LinearCombination factor, applicable if schedule:lincomb')
    parser.add_argument('--sampler', default='random', type=str, help='sampling, [random,upsampling,downsampling]')
    parser.add_argument('--reduction', default='mean', type=str, help='reduce mini batch')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None,
                        help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float,
                        help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
