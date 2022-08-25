_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="IIFLoss",variant='raw'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=0.0, override=dict(name='fc_cls')))))


work_dir='./experiments/iif/iif_r50_rfs_4x4_18e/'

evaluation = dict(interval=18, metric=['bbox', 'segm'])

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=18)
