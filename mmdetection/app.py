import streamlit as st
from pycocotools.coco import COCO
from mmdet.datasets import build_dataset
from mmdet.apis import init_detector, inference_detector
from mmcv import Config

import os
import cv2
import numpy as np
import copy

st.set_page_config(layout="wide")

root = "../../dataset/"

classes = {
    "General trash": 0,
    "Paper": 1,
    "Paper pack": 2,
    "Metal": 3,
    "Glass": 4,
    "Plastic": 5,
    "Styrofoam": 6,
    "Plastic bag": 7,
    "Battery": 8,
    "Clothing": 9,
    0: "General trash",
    1: "Paper",
    2: "Paper pack",
    3: "Metal",
    4: "Glass",
    5: "Plastic",
    6: "Styrofoam",
    7: "Plastic bag",
    8: "Battery",
    9: "Clothing",
}

colors = [
    (102, 147, 171),
    (247, 215, 105),
    (37, 94, 150),
    (241, 103, 66),
    (162, 217, 227),
    (241, 211, 203),
    (159, 186, 121),
    (213, 69, 132),
    (109, 127, 91),
    (214, 183, 159),
]
model_name = "cascade_rcnn_swint_fpn_1x_coco"
cfg_path = "./configs/_trash_/" + model_name + ".py"
cfg = Config.fromfile(cfg_path)


@st.cache(allow_output_mutation=True)
def load_data():
    ann_file = root + "train_split.json"
    origin_dataset = COCO(ann_file)
    aug_dataset = build_dataset(cfg.data.train)

    return origin_dataset, aug_dataset


# @st.cache(hash_funcs=MyModelClass: lambda model: model.get_config()})
def load_model():
    epoch = "latest"

    work_dir = "./work_dirs/" + model_name
    checkpoint_path = os.path.join(work_dir, f"{epoch}.pth")

    model = model = init_detector(cfg_path, checkpoint_path)  # build detector

    return model


def get_img_infos(dataset):
    imgIds = dataset.getImgIds()
    Imgs = [dataset.loadImgs(imgId)[0] for imgId in imgIds]
    return Imgs


def create_state(dataset, id):
    ann_ids = dataset.getAnnIds(imgIds=id)
    anns = dataset.loadAnns(ann_ids)

    state = {}
    state["labels"] = np.array([ann["category_id"] for ann in anns])
    state["boxes_id"] = np.array([ann["id"] for ann in anns])
    state["boxes"] = np.array([ann["bbox"] for ann in anns]).astype(int)
    state["viz"] = np.array([True for _ in range(len(state["boxes_id"]))])

    return state


def make_check_box(tab_bbox, labels, viz) -> None:
    label_buttons = {}

    for label in range(10):
        if label in labels:
            label_buttons[label] = tab_bbox.checkbox(classes[label], value=True)
        else:
            label_buttons[label] = tab_bbox.checkbox(classes[label], value=False, disabled=True)

    for i, label in enumerate(labels):
        if label_buttons[label]:
            viz[i] = True
        else:
            viz[i] = False


def make_multi_select_box(tab, boxes_id, viz) -> None:
    selected_id = [id for i, id in enumerate(boxes_id) if viz[i]]
    selected_boxes = tab.multiselect("Select want box_ids", boxes_id, selected_id)

    for i, id in enumerate(boxes_id):
        if id in selected_boxes and viz[i]:
            viz[i] = True
        else:
            viz[i] = False


def make_bbox(image, state):
    boxes, labels = state["boxes"], state["labels"]
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if state["viz"][i]:
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[label], 3)

    return image


def make_bbox_image(dataset, origin_image, id, tab_bbox):
    tab_bbox_col1, tab_bbox_col2 = tab_bbox.columns(2)

    state = create_state(dataset, id)

    make_check_box(tab_bbox_col2, state["labels"], state["viz"])
    make_multi_select_box(tab_bbox, state["boxes_id"], state["viz"])

    image = copy.deepcopy(origin_image)
    image = make_bbox(image, state)
    image = cv2.resize(image, (512, 512))

    tab_bbox_col1.image(image, caption="Selected Image")

    return image


def make_aug_image(aug_datasets, tab_augmentation, index, origin_image) -> None:
    tab_augmentation_col1, tab_augmentation_col2 = tab_augmentation.columns(2)

    aug_image = aug_datasets[index]["img"].data
    aug_image = aug_image.permute(1, 2, 0)
    aug_image = np.ascontiguousarray(aug_image.data.cpu())

    tab_augmentation_col1.image(origin_image, caption="Original Image")
    tab_augmentation_col2.image(aug_image, caption="Augmentation Image", clamp=True)


def make_pred_image(model, tab_predict, origin_image, bbox_image) -> None:
    tab_predict_col1, tab_predict_col2 = tab_predict.columns(2)

    output = inference_detector(model, origin_image)
    # predict_image = show_result_pyplot(model, origin_image, output, score_thr=0.5)
    predict_image = model.show_result(origin_image, output)
    predict_image = cv2.resize(predict_image, (512, 512))

    tab_predict_col1.image(bbox_image, caption="GT_Bboxes")
    tab_predict_col2.image(predict_image, caption="Predict_Bboxes")


def main():
    st.title("Trash Object Detection")

    if "dataset" not in st.session_state:
        st.session_state.dataset, st.session_state.aug_dataset = load_data()
        st.session_state.img_infos = get_img_infos(st.session_state.dataset)
        st.session_state.model = load_model()

    file_names = [img_info["file_name"] for img_info in st.session_state.img_infos]
    file_name = st.selectbox("Select Image", file_names, index=0)
    index = file_names.index(file_name)

    tab_bbox, tab_augmentation, tab_eda, tab_prediction = st.tabs(
        ["Image", "Augmentation", "EDA", "Predict"]
    )

    file_id = st.session_state.img_infos[index]["id"]
    origin_image = cv2.imread(os.path.join(root, file_name))

    bbox_image = make_bbox_image(st.session_state.dataset, origin_image, file_id, tab_bbox)
    make_aug_image(st.session_state.aug_dataset, tab_augmentation, index, origin_image)
    make_pred_image(st.session_state.model, tab_prediction, origin_image, bbox_image)


main()
