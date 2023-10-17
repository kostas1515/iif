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

<h1>Pretrained Models</h1>
<table style="float: center; margin-right: 10px;">
    <tr>
        <th>Method</th>
        <th>AP</th>
        <th>AP<sup>r</sup></th>
        <th>AP<sup>c</sup></th>
        <th>AP<sup>f</sup></th>
        <th>AP<sup>b</sup></th>
        <th>Model</th>
        <th>Output</th>
    </tr>
    <tr>
        <td>IIF_r50</td>
        <td>26.3</td>
        <td>18.6</td>
        <td>25.2</td>
        <td>30.8</td>
        <td>25.8</td>
        <td><a href="https://www.dropbox.com/scl/fi/ba2v7tb1d9lwyecesplnp/epoch_12.pth?rlkey=98vnto2dm0npu5a148b4twj4g&dl=0">weights</a></td>
        <td><a href="https://www.dropbox.com/scl/fi/4k6jyvblibolvcbpbwuw7/20220701_140348.log?rlkey=np2mzz3ycsnn5f4m0ge31xz66&dl=0">log</a>|<a href="https://www.dropbox.com/scl/fi/iu6h25c3hmcgxwsuwe120/fasa_iof_base10_giou16_r50_rfs_cos_norm_4x4_1x_rsb.py?rlkey=su3mfo8fn7np283x88w32z08k&dl=0">config</a></td>
    </tr>
    <tr>
        <td>IIF_r50_rsb</td>
        <td>27.4</td>
        <td>19.4</td>
        <td>26.8</td>
        <td>31.5</td>
        <td>27.4</td>
        <td><a href="https://drive.google.com/file/d/17djufBgqYNLq3_BksTT_XDw9cIunyxaa/view?usp=sharing">weights</a></td>
        <td><a href="https://drive.google.com/file/d/1IwL51w9cm5Kb2L861F1fijBp-xyMLghY/view?usp=sharing">log</a>|<a href="https://drive.google.com/file/d/1A7hl4FLKOAlQKqohVN_RZ3_4RsHn4xM_/view?usp=sharing">config</a></td>
    </tr>
</table>
<b>Note</b><p>The ResNet_rsb log-file shows mask AP 27.3 because it was tested with mask_thr_binary=0.5, which slightly lowers mask AP by 0.1. To get the same result like in the paper use mask_thr_binary=0.4 during inference.</p>
