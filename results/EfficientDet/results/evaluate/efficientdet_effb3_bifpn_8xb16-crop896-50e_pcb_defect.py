auto_scale_lr = dict(base_batch_size=4, enable=False)
backend_args = None
batch_augments = [
    dict(size=(
        896,
        896,
    ), type='BatchFixedSizePad'),
]
checkpoint = '/kaggle/input/checkpoint/efficientnet-b3_3rdparty_8xb32-aa-advprop_in1k_20220119-53b41118.pth'
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.EfficientDet.efficientdet',
    ])
data_root = './datasets/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evalute_type = 'CocoMetric'
image_size = 896
launcher = 'none'
load_from = '/kaggle/input/inference-efficientdet/best_coco_MP_precision_epoch_43.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 100
metainfo = dict(classes=(
    'MP',
    'OC',
    'SC',
    'SP',
    'SPC',
))
model = dict(
    backbone=dict(
        arch='b3',
        conv_cfg=dict(type='Conv2dSamePadding'),
        drop_path_rate=0.3,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint=
            '/kaggle/input/checkpoint/efficientnet-b3_3rdparty_8xb32-aa-advprop_in1k_20220119-53b41118.pth',
            prefix='backbone',
            type='Pretrained'),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        norm_eval=False,
        out_indices=(
            3,
            4,
            5,
        ),
        type='EfficientNet'),
    bbox_head=dict(
        anchor_generator=dict(
            center_offset=0.5,
            octave_base_scale=4,
            ratios=[
                1.0,
                0.5,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=160,
        in_channels=160,
        loss_bbox=dict(beta=0.1, loss_weight=50, type='HuberLoss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=1.5,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_classes=5,
        num_ins=5,
        stacked_convs=4,
        type='EfficientDetSepBNHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            48,
            136,
            384,
        ],
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_stages=6,
        out_channels=160,
        start_level=0,
        type='BiFPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(
            iou_threshold=0.3,
            method='gaussian',
            min_score=0.001,
            sigma=0.5,
            type='soft_nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='EfficientDet')
norm_cfg = dict(eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.16, momentum=0.9, type='SGD', weight_decay=4e-05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=917, start_factor=0.1, type='LinearLR'),
    dict(
        T_max=49,
        begin=1,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_min=0.0,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='./datasets/',
        metainfo=dict(classes=(
            'MP',
            'OC',
            'SC',
            'SP',
            'SPC',
        )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='./datasets/val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_batch_size_per_gpu = 2
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='./datasets/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=(
            'MP',
            'OC',
            'SC',
            'SP',
            'SPC',
        )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 1
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='./datasets/',
        metainfo=dict(classes=(
            'MP',
            'OC',
            'SC',
            'SP',
            'SPC',
        )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='./datasets/val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_num_workers = 1
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'results/evaluate'
