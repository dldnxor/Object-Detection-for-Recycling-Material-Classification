# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_scale = (1024, 1024)
img_norm_cfg = dict(mean=[123.65067045, 117.3973029, 110.0754267], std=[54.034579050000005, 53.36968695, 54.78390675], to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', crop_size=(0.7,0.7),
                 crop_type='relative_range',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True),
    dict(type='Mosaic', img_scale=img_scale, center_ratio_range=(0.9, 1.1), pad_val=114.0, prob=0.6),
    # dict(
        # type='RandomAffine',
        # scaling_ratio_range=(0.1, 2),
        # border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train_split_NEW.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
    ),

    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_split_NEW.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")
