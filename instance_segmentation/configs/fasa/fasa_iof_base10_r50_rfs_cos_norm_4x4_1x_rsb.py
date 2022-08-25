_base_ = ['./fasa_iof_base10_r50_rfs_cos_norm_4x4_2x.py'
]

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedDynamicRunner', max_epochs=12)
evaluation = dict(metric=['bbox', 'segm'], interval=12)


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0002,
    weight_decay=0.05,
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))

work_dir = 'experiments/fasa_iof_base10_r50_rfs_cos_norm_4x4_1x_rsb'
# work_dir = 'experiments/test'