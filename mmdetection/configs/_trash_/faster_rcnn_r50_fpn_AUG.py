_base_ = [
    './_base_/datasets/hyunsoo_custom_dataset.py',
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/schedules/schedule_1x_SGD.py',
    './_base_/default_runtime_autolr.py',
]