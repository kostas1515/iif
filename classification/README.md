# Image classification reference training and testing scripts 

This folder contains reference training and testing scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

## CIFAR-LT 
For CIFAR-LT datasets, all models have been trained on 4x V100 GPUs with 
the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--model`                |`se_resnet32`|
| `--dset_name`            |`cifar{100}`|
| `--data_path`            |`../../../datasets/`|
| `--classif_norm`         |`cosine`|
| `--batch_size`           | `16`   |
| `--epochs`               | `400`  |
| `--lr`                   | `0.1`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--milestones  `         |`360 380` |
| `--lr-gamma`             | `0.01` |
| `--imb_type`             | `exp`  |
| `--imb_factor`           | `0.01` |
| `--mixup`                | `0.2`  |
| `--auto-augment`         | `cifar`|


### CIFAR100-LT representation learning
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=cifar100 --imb_factor 0.01 --model se_resnet32 --output-dir ../checkpoints/se_r32_c100_ce_mean -b 16 --reduction mean --lr 0.1 --milestones 360 380 --epochs 400 --wd 0.0001 --lr-gamma 0.01 --classif_norm cosine --mixup 0.2 --auto-augment cifar
```

### CIFAR100-LT classifier learning with IIF
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=cifar100 --imb_factor 0.01 --model se_resnet32 --output-dir ../checkpoints/se_r32_c100_ce_mean_decoup -b 16 --reduction mean --epochs 20 --wd 0.0001 --classif_norm cosine --auto-augment cifar --load_from ../checkpoints/se_r32_c100_ce_mean/model_399.pth --lr 0.0001 --classif iif --iif smooth --decoup
```

### CIFAR100-LT post-process IIF
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=cifar100 --imb_factor 0.01 --model se_resnet32 -b 16 --classif_norm cosine --load_from ../checkpoints/se_r32_c100_ce_mean/model_399.pth --classif iif --iif smooth --test-only
```

## ImageNet-LT 
For ImageNet-LT dataset, all models have been trained on 4x V100 GPUs with 
the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--model`                |`se_resnet50`|
| `--dset_name`            |`imagenet_lt`|
| `--data_path`            |`../../../datasets/ILSVRC/Data/CLS-LOC/`|
| `--classif_norm`         |`cosine`|
| `--batch_size`           | `64`   |
| `--epochs`               | `200`  |
| `--lr`                   | `0.2`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--cosine_scheduler`     | `True` |
| `--mixup`                | `0.2`  |
| `--auto-augment`         | `imagenet`|


### ImageNet-LT representation learning
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=imagenet_lt --model se_resnet50 --output-dir ../checkpoints/se_r50_ilt_ce_mean -b 64 --cosine_scheduler --reduction mean --lr 0.2 --epochs 200 --classif_norm cosine --mixup 0.2 --auto-augment imagenet
```

### ImageNet-LT classifier learning with IIF
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=imagenet_lt --model se_resnet50 --output-dir ../checkpoints/se_r50_ilt_ce_mean_decoup -b 64 --reduction mean --lr 0.00002 --epochs 5 --classif_norm cosine --mixup 0.2 --auto-augment imagenet --load_from ../checkpoints/se_r50_ilt_ce_mean/model_199.pth --classif iif --decoup --iif smooth
```

### ImageNet-LT post-process IIF
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=imagenet_lt --model se_resnet50 - -b 64 --classif_norm cosine --mixup 0.2 --load_from ../checkpoints/se_r50_ilt_ce_mean/model_199.pth --classif iif --iif smooth --test-only
```

### Places-LT representation learning
For Places-LT dataset, first pretrain SE-ResNet152 on full ImageNet. The hyper-parameters are adjusted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnext101-32x4d/README.md. Use the following command for pretraining:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=ImageNet --model se_resnet152 --output-dir ../checkpoints/imagenet_full_se_r152_cosine_scheduler_mixup -b 64 --cosine_scheduler --reduction mean --lr 0.256 --epochs 100 --classif_norm cosine --mixup 0.2 --momentum 0.875 --wd 6.103515625e-05
```

Then freeze all parameters except for final Resnet-block and Classifier and train for 30 epochs using:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --dset_name places_lt --data-path ../../../datasets/places365_standard/ --model se_resnet152 --epochs 30 -b 64 --lr 0.2 --output-dir ../checkpoints/places_se_r152_softmax_e30 --cosine_scheduler --pretrained ../checkpoints/imagenet_full_se_r152_cosine_scheduler_mixup/model_99.pth --mixup 0.2 --auto-augment randaugment --classif_norm lr_cosine
```

### Places-LT classifier learning with IIF
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=places_lt --data-path ../../../datasets/places365_standard/ --model se_resnet152 --output-dir ../checkpoints/places_se_r152_softmax_e30 -b 64 --reduction mean --lr 0.00002 --epochs 5 --classif_norm cosine --mixup 0.2 --auto-augment imagenet --load_from ../checkpoints/places_se_r152_softmax_e30/model_29.pth --classif iif --decoup --iif smooth
```

### Places-LT post-process IIF
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=places_lt --data-path ../../../datasets/places365_standard/ --model se_resnet152  -b 64 --classif_norm lr_cosine --load_from ../checkpoints/places_se_r152_softmax_e30/model_29.pth --classif iif --iif smooth --test-only
```