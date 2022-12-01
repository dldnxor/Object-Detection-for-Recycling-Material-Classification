dataset_type = 'CocoDataset'
data_root = '../../dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
           'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
img_norm_cfg = dict(
    mean=[123.65067045, 117.3973029, 110.0754267],
    std=[54.034579050000005, 53.36968695, 54.78390675],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(0.5, 0.5),
        crop_type='relative_range',
        allow_negative_crop=False,
        recompute_bbox=True,
        bbox_clip_border=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(
        type='Normalize',
        mean=[123.65067045, 117.3973029, 110.0754267],
        std=[54.034579050000005, 53.36968695, 54.78390675],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.65067045, 117.3973029, 110.0754267],
                std=[54.034579050000005, 53.36968695, 54.78390675],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='../../dataset/train_split_NEW.json',
        img_prefix='../../dataset/',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(0.5, 0.5),
                crop_type='relative_range',
                allow_negative_crop=False,
                recompute_bbox=True,
                bbox_clip_border=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
            dict(
                type='Normalize',
                mean=[123.65067045, 117.3973029, 110.0754267],
                std=[54.034579050000005, 53.36968695, 54.78390675],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='../../dataset/val_split_NEW.json',
        img_prefix='../../dataset/',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.65067045, 117.3973029, 110.0754267],
                        std=[54.034579050000005, 53.36968695, 54.78390675],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='../../dataset/test.json',
        img_prefix='../../dataset/',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.65067045, 117.3973029, 110.0754267],
                        std=[54.034579050000005, 53.36968695, 54.78390675],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')