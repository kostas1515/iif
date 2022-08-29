<h1> Inverse Image Frequence for Long-tailed Image Recognition </h1>


### Progress

- [x] Training code.
- [x] Evaluation code.
- [x] LVIS v1.0, ImageNet-LT, Places-LT datasets.
- [ ] Provide checkpoint models.


<h1> Tested with </h1>
<div>
 <ul>
  <li>python==3.8.12</li>
  <li>torch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>mmdet==2.15.1</li>
  <li>lvis</li>
  <li>Tested on CUDA 10.1,10.0</li>
</ul> 
</div>
<b>Please Note that there is a reproducibility issue when using CUDA 10.2, as it drops classification performance by ~5%. For this reason please use either cuda 10.1 or cuda 10.0. Other versions are not tested.</b>

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
pip install catalyst
pip install imgaug
pip install randaugment
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

### Datasets
For COCO and LVIS datasets:
1. Create data directory, download COCO 2017 datasets at https://cocodataset.org/#download (2017 Train images [118K/18GB], 2017 Val images [5K/1GB], 2017 Train/Val annotations [241MB]) and extract the zip files:

```
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

#download and unzip LVIS annotations
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip

```

2. modify mmdetection/configs/_base_/datasets/lvis_v1_instance.py and make sure data_root variable points to the above data directory, e.g., data_root= "\<user_path\>"

For ImageNet and Places-LT:
1. Download the [ImageNet_2014](http://image-net.org/index) and [Places_365](http://places2.csail.mit.edu/download.html).