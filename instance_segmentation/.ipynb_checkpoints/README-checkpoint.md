<h1> Inverse Image Frequence for Long-tailed Instance Segmentation and Object detection </h1>

In this folder instance segmentation and object detection scripts are described.

<h1> Tested with </h1>
<div>
 <ul>
  <li>python==3.8.12</li>
  <li>torch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>mmdet==2.15.1</li>
  <li>lvis</li>
  <li>Tested on CUDA 10.2 and RHEL 8 system</li>
</ul> 
</div>


<h1> Getting Started </h1>
Create a virtual environment

```
conda create --name mmdet pytorch=1.7.1 -y
conda activate mmdet
```

1. Install dependency packages
```
conda install torchvision -y
conda install pandas scipy -y
conda install opencv -y
```

2. Install MMDetection
```
pip install openmim
mim install mmdet==2.15.1
```
3. Clone this repo
```
git clone https://github.com/kostas1515/iif.git
cd iif
```
4. Create data directory, download COCO 2017 datasets at https://cocodataset.org/#download (2017 Train images [118K/18GB], 2017 Val images [5K/1GB], 2017 Train/Val annotations [241MB]) and extract the zip files:

```
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

#download and unzip LVIS annotations
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip

```

5. modify mmdetection/configs/_base_/datasets/lvis_v1_instance.py and make sure data_root variable points to the above data directory, e.g., data_root= "\<user_path\>"


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

