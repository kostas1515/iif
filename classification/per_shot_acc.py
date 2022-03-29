import torch

import resnet_pytorch
import imbalanced_dataset 
import numpy as np
import utils
import argparse
import custom
from itertools import chain
from apex import amp

def main(args):
    auto_augment_policy = getattr(args, "auto_augment", None)
    dataset, dataset_test, _, test_sampler = imbalanced_dataset.get_imagenet_lt(False, root=args.root,
                              auto_augment=auto_augment_policy,sampler = args.sampler)
    num_classes = len(dataset.cls_num_list)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    model = eval(f'resnet_pytorch.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}")')
    

    if (args.classif== 'iif'):
        criterion = custom.IIFLoss(dataset,variant=args.iif,iif_norm=0,reduction='mean',weight=None)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load(args.load_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.to('cuda')
    if args.apex:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    avg_acc,preds,targets=evaluate(model.cuda(), criterion, data_loader_test, device='cuda')

    print(f'Avg Acc is: {avg_acc}')

    f,c,r = shot_acc(np.array(preds),np.array(targets),dataset.targets)

    print(f'Many shot Acc is: {f}, median shot Acc is: {c}, low shot Acc is: {r}')

    return 0



def shot_acc (preds, labels, train_targets, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    
    if isinstance(train_targets, np.ndarray):
        training_labels = np.array(train_targets).astype(int)
    else:
        training_labels = np.array(train_targets).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)



def evaluate(model, criterion, data_loader, device, print_freq=10):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    predictions=[]
    targets=[]
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
#             loss = criterion(output, target)
            if (type(output)==tuple):
                out=criterion(output,infer=True)
                acc1, acc5 = utils.accuracy(out, target, topk=(1, 5))
            else:
                if hasattr(criterion, 'iif'):
                    output=criterion(output,infer=True)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                
            predictions=list(chain(predictions,output.argmax(axis=1).tolist()))
            targets=list(chain(targets,target.tolist()))
            
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

    return metric_logger.acc1.global_avg,predictions,targets

def get_args_parser():
    parser = argparse.ArgumentParser(description='Parse arguments for per shot acc.')

    parser.add_argument(
        '--root', default='../../../datasets/ILSVRC/Data/CLS-LOC/', help='dataset')
    parser.add_argument('--auto-augment', default=None,
                        help='auto augment policy (default: None)')
    parser.add_argument('--sampler', default='random', type=str, help='sampling, [random,upsampling,downsampling]')
    parser.add_argument('--iif', default='raw',type=str, help='Type of IIF variant')
    parser.add_argument('--classif', default='ce',type=str, help='Type of classification')
    parser.add_argument('--classif_norm', default=None,type=str, help='Type of classifier Normalisation {None,norm,cosine')
    parser.add_argument('--load_from', default='', help='load wweights only from checkpoint')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 16)')
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O2', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )


    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    print('end of program')