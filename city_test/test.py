import cv2
import os.path as op
from glob import glob
import numpy as np
import json

import Config


def create_mask(mask):
    label = Config.DATA_INFO.LABEL
    seg_maps = {}
    for id in list(label.keys()):
        color = label[id]
        seg_map = np.all(mask == color, axis=-1)
        if seg_map.any():
            seg_maps[id] = seg_map
    return seg_maps


def extract_mask(id, mask):
    label = Config.DATA_INFO.LABEL
    color = label[id]
    colored_mask = mask.astype(np.uint8) * 255
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2BGR)
    colored_mask = colored_mask * color / 255
    colored_mask = np.array(colored_mask, dtype=np.uint8)
    return colored_mask


def visualize_mask(masks):
    result_mask = np.zeros(Config.DATA_INFO.IMAGE_SHAPE, np.uint8)
    for (id, mask) in (list(masks.items())):
        extracted = extract_mask(id, mask)
        result_mask = cv2.add(result_mask, extracted)
    cv2.imshow("mask", result_mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

sample_mask = op.join(Config.DATA_INFO.RAW_DATA_PATH, 'gtFine', 'gtFine', 'train', 'aachen')

imgs = glob(op.join(sample_mask, '*_color.png'))
imgs.sort()
for img in imgs:
    image = cv2.imread(img)
    cv2.imshow('image', image)
    mask = create_mask(image)
    visualize_mask(mask)

# annotations = glob(op.join(sample_mask, '*_polygons.json'))
# annotations.sort()
# for annotation in annotations:
#     with open(annotation, 'r') as f:
#         data = json.load(f)

