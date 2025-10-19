affine_scale = 0.5
backend_args = None
base_lr = 0.01
batch_shapes_cfg = dict(
    batch_size=2,
    extra_pad_ratio=0.5,
    img_size=640,
    size_divisor=32,
    type='BatchShapePolicy')
data_root = './datasets/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 1
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=100,
        scheduler_type='cosine',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
launcher = 'none'
load_from = '/kaggle/input/inference-yolov6/best_coco_MP_precision_epoch_23.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr_factor = 0.01
max_epochs = 100
max_keep_ckpts = 1
metainfo = dict(classes=(
    'MP',
    'OC',
    'SC',
    'SP',
    'SPC',
))
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        block_cfg=dict(
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            type='ConvWrapper'),
        deepen_factor=1,
        hidden_ratio=0.5,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=[
            1,
            2,
            3,
            4,
        ],
        type='YOLOv6CSPBep',
        use_cspsppf=True,
        widen_factor=1),
    bbox_head=dict(
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                128,
                256,
                512,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=5,
            reg_max=16,
            type='YOLOv6HeadModule',
            widen_factor=1),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='giou',
            loss_weight=2.5,
            reduction='mean',
            return_iou=False,
            type='IoULoss'),
        type='YOLOv6Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='ReLU'),
        block_act_cfg=dict(inplace=True, type='SiLU'),
        block_cfg=dict(
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            type='ConvWrapper'),
        deepen_factor=1,
        hidden_ratio=0.5,
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=12,
        out_channels=[
            128,
            256,
            512,
        ],
        type='YOLOv6CSPRepBiPAFPN',
        widen_factor=1),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=1,
            beta=6,
            num_classes=5,
            topk=13,
            type='BatchTaskAlignedAssigner'),
        initial_assigner=dict(
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=5,
            topk=9,
            type='BatchATSSAssigner'),
        initial_epoch=4),
    type='YOLODetector')
num_classes = 5
optim_wrapper = dict(
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=4,
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
resume = False
save_epoch_intervals = 1
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='val.json',
        batch_shapes_cfg=dict(
            batch_size=2,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
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
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='./datasets/val.json',
    classwise=True,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'train.json'
train_batch_size_per_gpu = 4
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1)
train_data_prefix = 'train/'
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        data_root='./datasets/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 1
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
        ),
        type='mmdet.PackDetInputs'),
]
val_ann_file = 'val.json'
val_batch_size_per_gpu = 2
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val/'
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='val.json',
        batch_shapes_cfg=dict(
            batch_size=2,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
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
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='./datasets/val.json',
    classwise=True,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_num_workers = 1
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.0005
widen_factor = 1
work_dir = 'results/evaluate'
