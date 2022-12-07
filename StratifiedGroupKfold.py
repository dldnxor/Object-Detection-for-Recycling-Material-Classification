import json
import funcy
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def filter_images(images, annotations):

    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)



# load json
annotation = '/opt/ml/dataset/train.json'


with open(annotation) as data:
    coco = json.load(data)
    info = coco['info']
    licenses = coco['licenses']
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']
    

var = [(ann['image_id'], ann['category_id']) for ann in annotations]
X = np.ones((len(annotations),1))
y = np.array([v[1] for v in var])    # category id
groups = np.array([v[0] for v in var])    # image id


# StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)



for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X,y, groups)):
    X_train = []
    y_train = []

    X_valid = []
    y_valid = []
    

    train_split_dataset_path = f'/opt/ml/dataset/train_split_fold{fold_ind+1}.json'
    valid_split_dataset_path = f'/opt/ml/dataset/val_split_fold{fold_ind+1}.json'
    

    for i in train_idx:
        X_train.append(annotations[i])
        y_train.append(y[i])    # 사실상 안쓰임
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)    # 사실상 안쓰임


    for i in val_idx:
        X_valid.append(annotations[i])
        y_valid.append(y[i])    # 사실상 안쓰임

    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)    # 사실상 안쓰임


    save_coco(train_split_dataset_path, info, licenses, filter_images(images, X_train), X_train.tolist(), categories)    # 새로운 coco dataset으로 저장
    print("TRAIN : Saved {} train entries in {}".format(len(X_train), train_split_dataset_path))

    save_coco(valid_split_dataset_path, info, licenses, filter_images(images, X_valid), X_valid.tolist(), categories)    # 새로운 coco dataset으로 저장
    print("VALID : Saved {} valid entries in {}".format(len(X_valid), valid_split_dataset_path))

    
