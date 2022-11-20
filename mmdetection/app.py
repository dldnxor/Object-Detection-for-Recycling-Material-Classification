import streamlit as st
from pycocotools.coco import COCO
from mmdet.datasets.pipelines import Compose, Albu
from configs._trash_._base_.datasets.transforms import train_pipeline, albu_train_transforms
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from mmdet.datasets import build_dataset
from mmcv import Config

st.set_page_config(layout='wide')

root='../../dataset/'

classes = {
    "General trash" : 0, "Paper" : 1, "Paper pack" : 2, "Metal" : 3, "Glass" : 4, 
           "Plastic" : 5, "Styrofoam" : 6, "Plastic bag" : 7, "Battery" : 8, "Clothing" : 9
           , 0 : "General trash", 1 : "Paper", 2 : "Paper pack", 3 : "Metal", 4 : "Glass", 
           5 : "Plastic", 6 : "Styrofoam", 7 : "Plastic bag", 8 : "Battery", 9 : "Clothing"}

colors = [(102, 147, 171), (247, 215, 105), (37, 94, 150),
 (241, 103, 66), (162, 217, 227), (241, 211, 203),
 (159, 186, 121), (213, 69, 132), (109, 127, 91), (214, 183, 159)]

cfg = Config.fromfile('./configs/_trash_/faster_rcnn_r50_fpn.py')



@st.cache
def load_data():
    ann_file = root + 'train.json'
    origin_dataset = COCO(ann_file)
    train_dataset = build_dataset(cfg.data.train)
    return origin_dataset, train_dataset


def get_img_infos(dataset):
    imgIds = dataset.getImgIds()
    Imgs = [dataset.loadImgs(imgId)[0] for imgId in imgIds]
    return Imgs


def create_state(dataset, id):
    ann_ids = dataset.getAnnIds(imgIds=id)
    anns = dataset.loadAnns(ann_ids)

    state = {}
    state['labels'] = np.array([ann['category_id'] for ann in anns])
    state['boxes_id'] = np.array([ann['id'] for ann in anns])
    state['boxes'] = np.array([ann['bbox'] for ann in anns]).astype(int)
    state['viz'] = np.array([True for _ in range(len(state['boxes_id']))])
    state['label_viz'] = np.array([True for _ in range(len(state['boxes_id']))])

    return state


def make_check_box(tab_bbox, labels, viz, label_viz) -> None:
    label_buttons = {}
    
    for label in range(10):
        if label in labels:
            if label_viz[label]:
                label_buttons[label] = tab_bbox.checkbox(classes[label], value=True, key=label)
            else:
                label_buttons[label] = tab_bbox.checkbox(classes[label], value=False, key=label)
        else:
            label_buttons[label] = tab_bbox.checkbox(classes[label], value=False, disabled=True, key=label)

    for i, label in enumerate(labels):
        if label_buttons[label]:
            viz[i] = True
            label_viz[i] = True
        else:
            viz[i] = False
            label_viz[i] = False


def make_multi_select_box(tab, boxes_id, viz) -> None:
    selected_id = [id for i, id in enumerate(boxes_id) if viz[i]]
    selected_boxes = tab.multiselect('Select want box_ids', boxes_id, selected_id)
    
    for i, id in enumerate(boxes_id):
        if id in selected_boxes and viz[i]:
            viz[i] = True
        else:
            viz[i] = False


def make_bbox(image, state):
    boxes, labels = state['boxes'], state['labels']
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if state['viz'][i] and state['label_viz'][i]:
            image = cv2.rectangle(image,
                (box[0], box[1]), (box[2], box[3]), colors[label], 3
            )

    return image


def make_bbox_image(dataset, image, id, tab_bbox):
    tab_bbox_col1, tab_bbox_col2 = tab_bbox.columns(2)

    state = create_state(dataset, id)

    make_check_box(tab_bbox_col2, state['labels'], state['viz'], state['label_viz'])
    make_multi_select_box(tab_bbox, state['boxes_id'], state['viz'])

    image = make_bbox(image, state)
    image = cv2.resize(image, (512, 512))

    tab_bbox_col1.image(image, caption='Selected Image')

    return image


def make_aug_image(aug_datasets, tab_augmentation, index, bbox_image) -> None:
    tab_augmentation_col1, tab_augmentation_col2 = tab_augmentation.columns(2)

    aug_image = aug_datasets[index]['img'].data
    aug_image = aug_image.permute(1,2,0)
    aug_image = np.ascontiguousarray(aug_image.data.cpu())
    
    # boxes = aug_datasets[index]['gt_bboxes'].data.numpy().astype(int)
    # labels = aug_datasets[index]['gt_labels'].data.numpy()

    # aug_image = make_bbox(aug_image, boxes, labels)
    tab_augmentation_col1.image(bbox_image, caption='Original Image')
    tab_augmentation_col2.image(aug_image, caption="Augmentation Image", clamp=True)


def main():
    st.title('Trash Object Detection')

    if 'dataset' not in st.session_state and 'img_infos' not in st.session_state:
        st.session_state.dataset, st.session_state.aug_dataset = load_data()
        st.session_state.img_infos = get_img_infos(st.session_state.dataset)

    file_names = [img_info['file_name'] for img_info in st.session_state.img_infos]
    file_name = st.selectbox('Select Image', file_names, index=0)
    index = file_names.index(file_name)

    tab_bbox, tab_augmentation, tab_prediction = st.tabs(["Image", "Augmentation", "Predict"])

    file_id = st.session_state.img_infos[index]['id']
    origin_image = cv2.imread(os.path.join(root, file_name))

    bbox_image = make_bbox_image(st.session_state.dataset, origin_image, file_id, tab_bbox)
    make_aug_image(st.session_state.aug_dataset, tab_augmentation, file_id, bbox_image)


main()