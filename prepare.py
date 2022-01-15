import pandas as pd
import os
import ast
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import cv2
import swifter
from pandarallel import pandarallel

import sys
sys.path.insert(1, './ULAP')

from ULAP.GuidedFilter import GuidedFilter
from ULAP.backgroundLight import BLEstimation
from ULAP.depthMapEstimation import depthMap
from ULAP.depthMin import minDepth
from ULAP.getRGBTransmission import getRGBTransmissionESt
from ULAP.global_Stretching import global_stretching
from ULAP.refinedTransmissionMap import refinedtransmissionMap
from ULAP.sceneRadiance import sceneRadianceRGB

DATASET_PATH = f'tensorflow-great-barrier-reef/yolo_enhance'

TRAIN_IMAGES_PATH = f"{DATASET_PATH}/images/train"
VAL_IMAGES_PATH = f"{DATASET_PATH}/images/val"

TRAIN_LABELS_PATH = f"{DATASET_PATH}/labels/train"
VAL_LABELS_PATH = f"{DATASET_PATH}/labels/val"


if not os.path.exists(TRAIN_IMAGES_PATH):
    os.makedirs(TRAIN_IMAGES_PATH)
if not os.path.exists(VAL_IMAGES_PATH):
    os.makedirs(VAL_IMAGES_PATH)
if not os.path.exists(TRAIN_LABELS_PATH):
    os.makedirs(TRAIN_LABELS_PATH)
if not os.path.exists(VAL_LABELS_PATH):
    os.makedirs(VAL_LABELS_PATH)

tqdm.pandas()
pandarallel.initialize()
df = pd.read_csv("train-0.1.csv")


def num_boxes(annotations):
    annotations = ast.literal_eval(annotations)
    return len(annotations)

#df['num_bbox'] = df['annotations'].apply(lambda x: num_boxes(x))
#df = df[df.num_bbox > 0]

print("Starting")

def apply_ulap(source_path, dest_path):
    img = cv2.imread(source_path)
    
    blockSize = 9
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    DepthMap = depthMap(img)
    DepthMap = global_stretching(DepthMap)
    guided_filter = GuidedFilter(img, gimfiltR, eps)
    refineDR = guided_filter.filter(DepthMap)
    refineDR = np.clip(refineDR, 0,1)
    
    AtomsphericLight = BLEstimation(img, DepthMap) * 255

    d_0 = minDepth(img, AtomsphericLight)
    d_f = 8 * (DepthMap + d_0)
    transmissionB, transmissionG, transmissionR = getRGBTransmissionESt(d_f)

    transmission = refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
    
    
    cv2.imwrite(dest_path, sceneRadiance)


def copy_file(row):
    print(row.image_path)
    if row.is_train:
        if not os.path.exists(f'{TRAIN_IMAGES_PATH}/{row.image_id}.jpg'):
            apply_ulap(row.image_path, f'{TRAIN_IMAGES_PATH}/{row.image_id}.jpg')
    else:
        if not os.path.exists(f'{VAL_IMAGES_PATH}/{row.image_id}.jpg'):
            apply_ulap(row.image_path, f'{VAL_IMAGES_PATH}/{row.image_id}.jpg')
            
_ = df.parallel_apply(lambda row: copy_file(row), axis=1)

IMG_WIDTH, IMG_HEIGHT = 1280, 720

def get_yolo_format_bbox(img_w, img_h, box):
    w = box['width'] 
    h = box['height']
    
    if (bbox['x'] + bbox['width'] > 1280):
        w = 1280 - bbox['x'] 
    if (bbox['y'] + bbox['height'] > 720):
        h = 720 - bbox['y'] 
        
    xc = box['x'] + int(np.round(w/2))
    yc = box['y'] + int(np.round(h/2)) 

    return [xc/img_w, yc/img_h, w/img_w, h/img_h]
    

for index, row in tqdm(df.iterrows()):
    annotations = ast.literal_eval(row.annotations)
    bboxes = []
    for bbox in annotations:
        bbox = get_yolo_format_bbox(IMG_WIDTH, IMG_HEIGHT, bbox)
        bboxes.append(bbox)
        
    if row.is_train:
        file_name = f"{TRAIN_LABELS_PATH}/{row.image_id}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    else:
        file_name = f"{VAL_LABELS_PATH}/{row.image_id}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
    with open(file_name, 'w') as f:
        for i, bbox in enumerate(bboxes):
            label = 0
            bbox = [label]+bbox
            bbox = [str(i) for i in bbox]
            bbox = ' '.join(bbox)
            f.write(bbox)
            f.write('\n')
                

