# dataset settings
dataset_type = "CocoDataset"
data_root = "../../dataset/"

classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)
img_norm_cfg = dict(
    mean=[123.65067045, 117.3973029, 110.0754267],
    std=[54.034579050000005, 53.36968695, 54.78390675],
    to_rgb=True,
)
albu_train_transforms = [
    dict(type="Blur", blur_limit=4, p=0.5),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.15,
        contrast_limit=0.2,
        brightness_by_max=False,
        p=0.5,
    ),
    dict(
        type="ShiftScaleRotate",
        rotate_limit=90,
        p=0.5,
    ),
]
train_pipeline = [
    # dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Mosaic",
        img_scale=(1024, 1024),
        center_ratio_range=(0.9, 1.1),
        pad_val=114.0,
        prob=0.6,
    ),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
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
        type="MultiImageMixDataset",
        dataset=dict(
            type="CocoDataset",
            ann_file=data_root + "train_split.json",
            img_prefix=data_root,
            classes=classes,
            pipeline=[dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True)],
            filter_empty_gt=False,
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_split.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")
