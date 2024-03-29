_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]


model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="IIFLoss",variant='raw',num_classes=80,path='./coco_files/idf_91.csv'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=0.0, override=dict(name='fc_cls')))))

load_from='./experiments/coco/baselines/r50_4x4_1x/latest.pth'
selectp=1
work_dir='./experiments/coco/iif_decoup_r50_4x4_1x/'