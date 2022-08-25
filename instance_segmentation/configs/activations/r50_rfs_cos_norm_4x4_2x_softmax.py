_base_ = [
    '../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v1.py'
]


# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False),
                                         cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
                                         ),
                           mask_head=dict(predictor_cfg=dict(type='NormedConv2d', tempearture=20))))

work_dir='./experiments/baselines/r50_rfs_cos_norm_4x4_2x_softmax/'
# work_dir='./experiments/test/'
