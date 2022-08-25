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
python -m torch.distributed.launch --nproc_per_node=4 --use_env  train.py --dset_name=cifar100 --imb_factor 0.01 --model se_resnet32 --output-dir ../checkpoints/se_r32_c100_ce_mean_decoup -b 16 --reduction mean --epochs 20 --wd 0.0001 --classif_norm cosine --mixup 0.2 --auto-augment cifar --load_from ../checkpoints/se_r32_c100_ce_mean/model_399.pth --lr 0.0001 --classif iif --iif smooth --decoup
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

## Places-LT 
For ImageNet-LT dataset, all models have been trained on 4x V100 GPUs with 
the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--model`                |`se_resnet152`|
| `--dset_name`            |`imagenet_lt`|
| `--data_path`            |`../../../datasets/places365_standard/`|
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