_base_ = [
    "./_base_/datasets/coco_detection.py",
    "./_base_/models/retinanet_r50_fpn.py",
    "./_base_/default_runtime.py",
    "./_base_/schedules/schedule_1x.py",
]

cudnn_benchmark = True
# norm_cfg=dict(type="SyncBN", requires_grad=True, eps=1e-3, momentum=0.01)
norm_cfg = dict(type="BN", requires_grad=True)
custom_imports = dict(imports=["mmcls.models"], allow_failed_imports=False)
checkpoint = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth"  # noqa
model = dict(
    backbone=dict(
        _delete_=True,  # Delete the backbone field in _base_
        type="mmcls.TIMMBackbone",  # Using timm from mmcls
        model_name="efficientnet_b3",
        features_only=True,
        pretrained=True,
        out_indices=(2, 3, 4),
    ),
    neck=dict(
        _delete_=True,
        type="BIFPN",
        in_channels=[48, 136, 384],
        out_channels=160,
        num_outs=5,
        # strides=[8, 16, 32],
        start_level=0,
        end_level=-1,
        stack=6,
        norm_cfg=norm_cfg,
    ),
    bbox_head=dict(
        type="RetinaSepBNHead",
        num_classes=10,
        in_channels=160,
        feat_channels=160,
        num_ins=5,
        norm_cfg=norm_cfg,
    ),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)),
)

# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = (896, 896)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=img_size, ratio_range=(0.8, 1.2), keep_ratio=True),
    # dict(type="RandomCrop", crop_size=img_size),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=img_size),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=img_size),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
