# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pdb

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data.datasets import load_coco_json, register_coco_instances
import torchvision

_root = os.getenv("DETECTRON2_DATASETS", "/mnt/d/dataset/")

NYU40_CATEGORIES = [
    {"color": [174,199,232], "isthing": 0, "id": 1, "name": "wall"},
    {"color": [152, 223, 138], "isthing": 0, "id": 2, "name": "floor"},
    {"color": [31, 119, 180], "isthing": 1, "id": 3, "name": "cabinet"},
    {"color": [255, 187, 120], "isthing": 1, "id": 4, "name": "bed"},
    {"color": [188, 189, 34], "isthing": 1, "id": 5, "name": "chair"},
    {"color": [140, 86, 75], "isthing": 1, "id": 6, "name": "sofa"},
    {"color": [255, 152, 150], "isthing": 1, "id": 7, "name": "table"},
    {"color": [214, 39, 40], "isthing": 1, "id": 8, "name": "door"},
    {"color": [197, 176, 213], "isthing": 1, "id": 9, "name": "window"},
    {"color": [148, 103, 189], "isthing": 1, "id": 10, "name": "bookshelf"},
    {"color": [196, 156, 148], "isthing": 1, "id": 11, "name": "picture"},
    {"color": [23, 190, 207], "isthing": 1, "id": 12, "name": "counter"},
    {"color": [178, 76, 76], "isthing": 0, "id": 13, "name": "blinds"},
    {"color": [247, 182, 210], "isthing": 1, "id": 14, "name": "desk"},
    {"color": [66, 188, 102], "isthing": 1, "id": 15, "name": "shelves"},
    {"color": [219, 219, 141], "isthing": 0, "id": 16, "name": "curtain"},
    {"color": [140, 57, 197], "isthing": 1, "id": 17, "name": "dresser"},
    {"color": [202, 185, 52], "isthing": 1, "id": 18, "name": "pillow"},
    {"color": [51, 176, 203], "isthing": 1, "id": 19, "name": "mirror"},
    {"color": [200, 54, 131], "isthing": 1, "id": 20, "name": "floor mat"},
    {"color": [92, 193, 61], "isthing": 0, "id": 21, "name": "clothes"},
    {"color": [78, 71, 183], "isthing": 0, "id": 22, "name": "ceiling"},
    {"color": [172, 114, 82], "isthing": 1, "id": 23, "name": "books"},
    {"color": [255, 127, 14], "isthing": 1, "id": 24, "name": "refridgerator"},
    {"color": [91, 163, 138], "isthing": 1, "id": 25, "name": "television"},
    {"color": [153, 98, 156], "isthing": 1, "id": 26, "name": "paper"},
    {"color": [140, 153, 101], "isthing": 0, "id": 27, "name": "towel"},
    {"color": [158, 218, 229], "isthing": 0, "id": 28, "name": "shower curtain"},
    {"color": [100, 125, 154], "isthing": 1, "id": 29, "name": "box"},
    {"color": [178, 127, 135], "isthing": 1, "id": 30, "name": "whiteboard"},
    {"color": [120, 185, 128], "isthing": 1, "id": 31, "name": "person"},
    {"color": [146, 111, 194], "isthing": 1, "id": 32, "name": "night stand"},
    {"color": [44, 160, 44], "isthing": 1, "id": 33, "name": "toilet"},
    {"color": [112,128,144], "isthing": 1, "id": 34, "name": "sink"},
    {"color": [96,207,209], "isthing": 1, "id": 35, "name": "lamp"},
    {"color": [227,119,194], "isthing": 1, "id": 36, "name": "bathtub"},
    {"color": [213,92,176], "isthing": 1, "id": 37, "name": "bag"},
    {"color": [94,106,211], "isthing": 0, "id": 38, "name": "otherstructure"},
    {"color": [82,84,163], "isthing": 1, "id": 39, "name": "otherfurniture"},
    {"color": [100,85,144], "isthing": 0, "id": 40, "name": "otherprop"},
]

SPLITS = {}
SPLITS["coco_2017_train_entity"]  = ("coco/train2017", "coco/annotations/entity_train2017.json")
SPLITS["coco_2017_val_entity"]    = ("coco/val2017", "coco/annotations/entity_val2017.json")

def _get_coco_trans_meta():
 	oc2nc_map = {category['id']: [cid, category["isthing"], category["name"], category["color"]] for cid, category in enumerate(COCO_CATEGORIES)}
 	NEW_COCO_CATEGORIES = []
 	for key, value in oc2nc_map.items():
         new_info = {"id": value[0], "isthing": value[1], "name": value[2], "color": value[3]}
         NEW_COCO_CATEGORIES.append(new_info)
 	
 	thing_ids     = [k["id"] for k in NEW_COCO_CATEGORIES]
 	thing_colors  = [k["color"] for k in NEW_COCO_CATEGORIES]
 	thing_classes = [k["name"] for k in NEW_COCO_CATEGORIES]
 	thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
 	ret = {
		"thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
 	return ret


for key, (image_root, json_file) in SPLITS.items():
 	register_coco_instances(key,
		_get_coco_trans_meta(),
		os.path.join(_root, json_file) if "://" not in json_file else json_file,
		os.path.join(_root, image_root)
 	)


SPLITS={}
SPLITS['3rscan_train_entity'] = ('entity_3rscan/','entity_3rscan/annotations/entity_train.json')
SPLITS['3rscan_val_entity'] = ('entity_3rscan/','entity_3rscan/annotations/entity_val.json')

def _get_3rscan_trans_meta():
	oc2nc_map = {category['id']: [cid, category["isthing"], category["name"], category["color"]] for cid, category in enumerate(NYU40_CATEGORIES)}
	NEW_COCO_CATEGORIES = []
	for key, value in oc2nc_map.items():
		new_info = {"id": value[0], "isthing": value[1], "name": value[2], "color": value[3]}
		NEW_COCO_CATEGORIES.append(new_info)
	
	thing_ids     = [k["id"] for k in NEW_COCO_CATEGORIES]
	thing_colors  = [k["color"] for k in NEW_COCO_CATEGORIES]
	thing_classes = [k["name"] for k in NEW_COCO_CATEGORIES]
	thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
	ret = {
		"thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
	return ret

for key, (image_root, json_file) in SPLITS.items():
	register_coco_instances(key,
		_get_3rscan_trans_meta(),
		os.path.join(_root, json_file) if "://" not in json_file else json_file,
		os.path.join(_root, image_root)
	)