import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from torch.utils.data import dataloader
import torchvision
import imbalanced_dataset
import pandas as pd

import presets
import utils
import custom
import resnet_cifar
import resnet_pytorch
import numpy as np
import initialisers

try:
    from apex import amp
except ImportError:
    amp = None


def record_result(result,args):
    df = pd.DataFrame.from_dict(vars(args))
    df['acc']=result
    file2save=os.path.join(args.output_dir,'results.csv')
    df = df.iloc[1: , :]
    if os.path.exists(file2save):
        df.to_csv(file2save, mode='a', header=False)
    else:
        df.to_csv(file2save)
    
    

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
    if epoch <1:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        
    if args.mixup is not None:
        mixup = custom.Mixup(criterion,alpha=args.mixup)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        
        if args.mixup is not None:
            image, targets_a, targets_b, lam = mixup(image, target)
        output = model(image)

        if args.mixup is not None:
            loss = mixup.mixup_criterion(output,targets_a, targets_b, lam)
        else:
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
            if hasattr(criterion, 'iif'):
                output=criterion(output,infer=True)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            
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



def select_training_param(model):
#     print(model)
    for v in model.parameters():
        v.requires_grad = False
    try:
        torch.nn.init.xavier_uniform_(model.linear.weight)
        model.linear.weight.requires_grad = True
        try:
            model.linear.bias.data.fill_(0.01)
            model.linear.bias.requires_grad = True
        except torch.nn.modules.module.ModuleAttributeError:
            pass
    except torch.nn.modules.module.ModuleAttributeError:
        torch.nn.init.xavier_uniform_(model.fc.weight)
        try:
            model.fc.bias.requires_grad = True
            model.fc.bias.data.fill_(0.01)
        except torch.nn.modules.module.ModuleAttributeError:
            pass
        model.fc.weight.requires_grad = True
        

    return model


def finetune_places(model):
    for v in model.parameters():
        v.requires_grad = False


    torch.nn.init.xavier_uniform_(model.fc.weight)
    try:
        model.fc.bias.requires_grad = True
        model.fc.bias.data.fill_(0.01)
    except torch.nn.modules.module.ModuleAttributeError:
        pass
    model.fc.weight.requires_grad = True
    
    for v in model.layer4[-1].parameters():
        v.requires_grad = True
        

    return model


def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(args)
    

    dataset,num_classes,data_loader,data_loader_test,train_sampler= initialisers.get_data(args)
    print("Creating model")
    try:
        model = eval(f'resnet_pytorch.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}",pretrained="{args.pretrained}")')
    except AttributeError:
        #model does not exist in pytorch load it from resnet_cifar
        model = eval(f'resnet_cifar.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}")')
            
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    criterion=initialisers.get_criterion(args,dataset,model,num_classes)
    

    opt_name = args.opt.lower()
    model_parameters=model.parameters()

    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'nesterov':
        optimizer = torch.optim.SGD(
            model_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=True)
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

#     if args.dset_name == 'places_lt':
#         model = finetune_places(model)
        
    if args.decoup:
        model = select_training_param(model)
        
    if args.cosine_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,args.epochs,0)
    else:
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

    if args.load_from:
        checkpoint = torch.load(args.load_from, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])


    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return
    

    print("Start training")
    start_time = time.time()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
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
    if args.record_result:
        if utils.is_main_process():
            record_result(best_acc,args)
    

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
    parser.add_argument('--model', default='resnet32', help='model,[resnet32,se_resnet32,resnet50,se_resnet50,resnext50_32x4d,se_resnext50_32x4d]')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--cosine_scheduler',
                        help='cosine scheduler',action='store_true')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--milestones',nargs='+', default=[360,380],type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100,
                        type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--load_from', default='', help='load weights only from checkpoint')
    parser.add_argument('--classif', default='ce',type=str, help='Type of classification')
    parser.add_argument('--classif_norm', default=None,type=str, help='Type of classifier Normalisation {None,norm,cosine')
    parser.add_argument('--gamma', default=0.0,type=float, help='Focal loss gamma hp')
    parser.add_argument('--alpha', default=None,type=float, help='Focal loss alpha hp')
    parser.add_argument('--iif', default='raw',type=str, help='Type of IIF variant- applicable if classif iif')
    parser.add_argument('--iif_norm', default=0, type=int, help='IIF norm')
    parser.add_argument('--decoup',action="store_true", help='Freeze all layers except classif layer')
    parser.add_argument('--mixup', default=None, type=float,
                        help='Mixup factor')
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
        help="Use pre-trained models from the modelzoo or local takes either values--> [pytorch, path/to/model]",
        default=None, type=str,
    )
    parser.add_argument(
        "--deffered",
        help="Use deferred schedule",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None,
                        help='auto augment policy (default: None)')

    parser.add_argument('--random-erase', default=0.0, type=float,
                        help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O2', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--record-result', dest="record_result",
        help="Record result in csv format",
        action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
