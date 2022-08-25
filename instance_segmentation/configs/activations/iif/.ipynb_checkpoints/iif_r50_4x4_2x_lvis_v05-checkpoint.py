_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py'
]

data = dict(train=dict(oversample_thr=0.0))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="IIFLoss",variant='raw',num_classes=1230,path='./lvis_files/idf_1231.csv'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=0.0, override=dict(name='fc_cls')))))

work_dir='./experiments/iif/iif_r50_4x4_2x_lvis_v05/'
# work_dir='./experiments/test/'