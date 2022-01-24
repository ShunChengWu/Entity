import os
import copy
import mmcv
import numpy as np
import pdb
import pycocotools.mask as mask_utils
# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
# from entityseg.data.datasets import NYU40_CATEGORIES
from datetime import date
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--folder','-f', type=str, 
                    default='/media/sc/SSD1TB/dataset/3RScan', 
                    help='directroy to 3RScan', required=True)
parser.add_argument('--mode','-m', type=str, choices=['train','val'],
                    default='train', 
                    help='mode. can be trian or val', required=True)
parser.add_argument('--folder','-f', type=str, 
                    default='/media/sc/SSD1TB/dataset/3RScan/', 
                    help='directroy to 3RScan', required=True)
parser.add_argument('--output','-o', type=str, 
                    default='/media/sc/SSD1TB/dataset/entity_3rscan/', 
                    help='directroy to 3RScan', required=True)
args = parser.parse_args()

def loadtext(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

prefix = args.mode
base_path =  args.output#"/media/sc/SSD1TB/dataset/entity_3rscan/"
path_3rscan = args.folder#'/media/sc/SSD1TB/dataset/3RScan'

# entity_base_path   = os.path.join(base_path, "entity_{}".format(prefix))
annotation_path               = os.path.join(base_path, "annotations/instances_{}.json".format(prefix))
save_thing_path               = os.path.join(base_path, "annotations/entity_thing_{}.json".format(prefix))
save_stuff_path               = os.path.join(base_path, "annotations/entity_stuff_{}.json".format(prefix))
save_entity_path              = os.path.join(base_path, "annotations/entity_{}.json".format(prefix))
path_split = os.path.join(path_3rscan,'splits')

instance_annotations       = mmcv.load(annotation_path)
instance_annotations_thing = copy.deepcopy(instance_annotations)
instance_annotations_stuff = copy.deepcopy(instance_annotations)

# update category
# categories_list               = NYU40_CATEGORIES
# catid_map                     = {category['id']: [cid, category["isthing"], category["name"]] for cid, category in enumerate(categories_list)}
# idcat_map = {}
# for key, value in catid_map.items():
#     idcat_map[value[0]] = [key,value[1]]
# print("Updating categories...")
# instance_annotations_thing["categories"] = []
# instance_annotations_stuff["categories"] = []
# for origin_catid, new_catid_info in catid_map.items():
#  	new_catid = new_catid_info[0]
#  	is_thing  = new_catid_info[1]
#  	name      = new_catid_info[2]
#  	if is_thing:
#          instance_annotations_thing["categories"].append({"id": new_catid, "name": name})
#  	else:
#          instance_annotations_stuff["categories"].append({"id": new_catid, "name": name})
# print("Update category finished")

# update annotations
instance_annotations_thing["annotations"] = []
instance_annotations_stuff["annotations"] = []

'''read list'''
scan_list = loadtext(os.path.join(path_split,prefix+'.txt'))
folder_list = os.listdir(os.path.join(base_path,prefix))
scan_list = list(set(scan_list).intersection(folder_list))

thing_id  = 0
stuff_id  = 0
# c_img_id = 0
# for scan_id in scan_list:
    # print('process scan',scan_id)
if True:
    npz_names = sorted(os.listdir(os.path.join(base_path, prefix)))
    for index, npz_name in enumerate(npz_names):
        print('\tnpz_name',npz_name)
        entity_info = np.load(os.path.join(base_path, prefix, npz_name))
        image_id = int(npz_name.split(".")[0])
        bounding_boxes = entity_info["bounding_box"]
        entity_id_map = entity_info["map"]
        entity_id_map = entity_id_map[0]
        if len(bounding_boxes)==0:
            continue
        # image_id = c_img_id
        print('\timage_id',image_id)
        # 0-x1, 1-y1, 2-x2, 3-y2, 4-category, 5-thing_or_stuff, 6-entity_id
        thing_mask  = bounding_boxes[:,5] > 0
        stuff_mask  = bounding_boxes[:,5] == 0
    
    	# begin thing
        thing_boxes = bounding_boxes[thing_mask]
        for thing_box in thing_boxes:
            x1, y1, x2, y2, category_id, thing_or_stuff, entity_id = thing_box
            area = (y2-y1) * (x2-x1)
            if "val" in prefix:
                mask = (entity_id_map==entity_id)
                mask = np.array(mask, order="F", dtype="uint8")
                rle  = mask_utils.encode(mask)
                rle["counts"] = rle["counts"].decode("utf-8")
    
            anno = {"iscrowd": 0, 
                    "area": area, 
                    "image_id": image_id, 
                    "bbox": [x1, y1, x2-x1, y2-y1], 
                    "category_id": category_id, 
                    "id": thing_id}
            if "val" in prefix:
                anno["segmentation"]=rle
            
            instance_annotations_thing["annotations"].append(anno)
            thing_id = thing_id + 1
        
        # begin stuff
        stuff_boxes = bounding_boxes[stuff_mask]
        for stuff_box in stuff_boxes:
            x1, y1, x2, y2, category_id, thing_or_stuff, entity_id = stuff_box
            area = (y2-y1) * (x2-x1)
            if "val" in prefix:
                mask = (entity_id_map==entity_id)
                mask = np.array(mask, order="F", dtype="uint8")
                rle  = mask_utils.encode(mask)
                rle["counts"] = rle["counts"].decode("utf-8")
    
            anno = {"iscrowd": 0, 
                    "area": area, 
                    "image_id": image_id, 
                    "bbox": [x1, y1, x2-x1, y2-y1], 
                    "category_id": category_id, 
                    "id": stuff_id}
            if "val" in prefix:
                anno["segmentation"]=rle
    
            instance_annotations_stuff["annotations"].append(anno)
            stuff_id = stuff_id + 1
        # c_img_id += 1
        print("{},{}".format(index, npz_name))
    pass

mmcv.dump(instance_annotations_thing, save_thing_path)
mmcv.dump(instance_annotations_stuff, save_stuff_path)

thing_info = instance_annotations_thing
stuff_info = instance_annotations_stuff

thst       = thing_info
thst['categories'] = instance_annotations['categories']
# thst["categories"].extend(stuff_info["categories"])
nums = len(thst["annotations"]) + 1
for index, anno in enumerate(stuff_info["annotations"]):
	anno["id"] = index + nums
	thst["annotations"].append(anno)
mmcv.dump(thst, save_entity_path)
