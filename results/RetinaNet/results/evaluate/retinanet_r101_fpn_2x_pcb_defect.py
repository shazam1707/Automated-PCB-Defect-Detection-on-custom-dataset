auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/kaggle/working/mmdetection/datasets/'
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
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/kaggle/input/inference-retiinanet/best_coco_MP_precision_epoch_16.pth'
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
        depth=101,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet101', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
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
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=5,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='/kaggle/working/mmdetection/datasets/',
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
    ann_file='/kaggle/working/mmdetection/datasets/val.json',
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
train_batch_size_per_gpu = 4
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='/kaggle/working/mmdetection/datasets/',
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
val_batch_size_per_gpu = 2
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='/kaggle/working/mmdetection/datasets/',
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
    ann_file='/kaggle/working/mmdetection/datasets/val.json',
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
