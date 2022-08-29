<h1> Inverse Image Frequence for Long-tailed Instance Segmentation and Object detection </h1>

In this folder instance segmentation and object detection scripts are described.

<h1>Training</h1>
To Train on multiple GPUs use <i>tools/dist_train.sh</i> to launch training on multiple GPUs:

```
./tools/dist_train.sh ./configs/<experiment>/<variant.py> <#GPUs>
```

E.g: To train IOF base10 with all enhancements on 4 GPUs use:
```
./tools/dist_train.sh ./configs/fasa/fasa_iof_base10_r50_rfs_cos_norm_4x4_2x.py 4
```

<h1>Testing</h1>

To test IIF:
```
./tools/dist_test.sh ./experiments/fasa_iof_base10_r50_rfs_cos_norm_4x4_2x/fasa_iof_base10_r50_rfs_cos_norm_4x4_2x.py ./experiments/fasa_iof_base10_r50_rfs_cos_norm_4x4_2x/latest.pth 4 --eval bbox segm
```


<h1>Reproduce</h1>
To reproduce the different IIF variants run:

```
./tools/dist_train.sh ./configs/activations/iif/variants/iif_<variant>_r50_4x4_1x.py 4
```

