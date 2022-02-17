from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import numpy as np
import pdb
import mmcv
import copy
import cv2
from collections import OrderedDict
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from panopticapi.utils import IdGenerator, rgb2id
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import json
from matplotlib import pyplot as plt

from tqdm import tqdm

from datetime import date

MIT_license = "MIT License Copyright (c) 2020 Johanna Wald"

def plot_np(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()

def loadtext(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def getLabelNames(path):
    import csv
    Scan3R=dict()
    NYU40=dict()
    Eigen13=dict()
    RIO27=dict()
    RIO7=dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            if not row[0].isnumeric():
                continue
            Scan3R[int(row[0])]=row[1]
            NYU40[int(row[2])] = row[3]
            Eigen13[int(row[4])] = row[5]
            RIO27[int(row[6])] = row[7]
            RIO7[int(row[8])] = row[9]
            # break
    return dict(sorted(Scan3R.items())),dict(sorted(NYU40.items())),\
           dict(sorted(Eigen13.items())),dict(sorted(RIO27.items())),dict(sorted(RIO7.items()))
           
def getLabelNameMapping(path):
    """
    return  toNameNYU40,toNameEigen,toNameRIO27,ttoNameRIO7
    """
    import csv
    raw=dict()
    toNameNYU40=dict()
    toNameEigen=dict()
    toNameRIO27=dict()
    toNameRIO7 =dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if not row[0].isnumeric():
                continue
            raw[row[1]]=row[1]
            toNameNYU40[row[1]] = row[3] if row[3] != '-' else 'none'
            toNameEigen[row[1]] = row[5] if row[5] != '-' else 'none'
            toNameRIO27[row[1]] = row[7] if row[7] != '-' else 'none'
            toNameRIO7[row[1]] = row[9] if row[9] != '-' else 'none'
    return raw, toNameNYU40,toNameEigen,toNameRIO27,toNameRIO7
def getLabelIdxMapping(path):
    """
    return  toNYU40,toEigen,toRIO27,toRIO7
    """
    import csv
    raw=dict()
    toNYU40=dict()
    toEigen=dict()
    toRIO27=dict()
    toRIO7 =dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            if not row[0].isnumeric():
                continue
            raw[int(row[0])] = int(row[0])
            toNYU40[int(row[0])] = int(row[2])
            toEigen[int(row[0])] = int(row[4])
            toRIO27[int(row[0])] = int(row[6])
            toRIO7[int(row[0])] = int(row[8])
            # break
    return raw,toNYU40,toEigen,toRIO27,toRIO7
def getLabelMapping(label_type:str,pth_mapping:str = ""):
    Scan3R528, NYU40,Eigen13,RIO27,RIO7 = getLabelNames(pth_mapping)
    NameScan3R528, toNameNYU40,toNameEigen13,toNameRIO27,toNameRIO7=getLabelNameMapping(pth_mapping)
    IdxScan3R528, toNYU40,toEigen13,toRIO27,toRIO7 = getLabelIdxMapping(pth_mapping)
    label_names=None
    label_name_mapping=None
    label_id_mapping=None
    label_type=label_type.lower()
    if label_type == 'nyu40':
        label_names = NYU40
        label_name_mapping = toNameNYU40
        label_id_mapping = toNYU40
    return label_names, label_name_mapping, label_id_mapping

def load_semseg(json_file, name_mapping_dict=None, mapping = True):    
    '''
    Create a dict that maps instance id to label name.
    If name_mapping_dict is given, the label name will be mapped to a corresponding name.
    If there is no such a key exist in name_mapping_dict, the label name will be set to '-'

    Parameters
    ----------
    json_file : str
        The path to semseg.json file
    name_mapping_dict : dict, optional
        Map label name to its corresponding name. The default is None.
    mapping : bool, optional
        Use name_mapping_dict as name_mapping or name filtering.
        if false, the query name not in the name_mapping_dict will be set to '-'
    Returns
    -------
    instance2labelName : dict
        Map instance id to label name.

    '''
    instance2labelName = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for segGroups in data['segGroups']:
            # print('id:',segGroups["id"],'label', segGroups["label"])
            # if segGroups["label"] == "remove":continue
            labelName = segGroups["label"]
            if name_mapping_dict is not None:
                if mapping:
                    if not labelName in name_mapping_dict:
                        labelName = 'none'
                    else:
                        labelName = name_mapping_dict[labelName]
                else:
                    if not labelName in name_mapping_dict.values():
                        labelName = 'none'

            instance2labelName[segGroups["id"]] = labelName.lower()#segGroups["label"].lower()
    return instance2labelName

def get_nyu40_labels():
    return [
    'wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'blinds',
'desk',
'shelves',
'curtain',
'dresser',
'pillow',
'mirror',
'floor mat',
'clothes',
'ceiling',
'books',
'refridgerator',
'television',
'paper',
'towel',
'shower curtain',
'box',
'whiteboard',
'person',
'night stand',
'toilet',
'sink',
'lamp',
'bathtub',
'bag',
'otherstructure',
'otherfurniture',
'otherprop',
]

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--folder','-f', type=str, 
                    default='/media/sc/SSD1TB/dataset/3RScan', 
                    help='directroy to 3RScan', required=True)
parser.add_argument('--mode','-m', type=str, choices=['train','val'],
                    default='train', 
                    help='mode. can be trian or val', required=True)
parser.add_argument('--output','-o', type=str, 
                    default='/media/sc/SSD1TB/dataset/entity_3rscan/', 
                    help='directroy to 3RScan', required=True)
parser.add_argument('--mapping', type=str, 
                    default='../data/3RScan.v2 Semantic Classes - Mapping.csv', 
                    help='directroy to 3RScan', required=False)
parser.add_argument('--list','-l', type=str, 
                    default='', 
                    help='the list of scan ids to be processed', required=True)
args = parser.parse_args()


# thread_num   = int(sys.argv[1])
# thread_idx   = int(sys.argv[2])
# type_        = sys.argv[3]

prefix = args.mode# "train"
# prefix = "val"
path_3rscan = args.folder#'/media/sc/SSD1TB/dataset/3RScan'
path_label_mapping = args.mapping#'/home/sc/research/Entity/Entity/EntitySeg/data/3RScan.v2 Semantic Classes - Mapping.csv'
pth_out = args.output# '/media/sc/SSD1TB/dataset/entity_3rscan'

path_data = os.path.join(path_3rscan, 'data/3RScan')
path_split = os.path.join(path_3rscan,'splits')
name_format_img = 'frame-{0:06d}.color.jpg'
name_format_inst = 'frame-{0:06d}.rendered.instances.png'
name_format_label = 'frame-{0:06d}.rendered.labels.png'
name_format_output = '{0:06d}'
name_semseg = 'semseg.v2.json'
nyu40_labels = get_nyu40_labels()

# train_list = loadtext(os.path.join(path_split,'train.txt'))
# val_list = loadtext(os.path.join(path_split,'val.txt'))
# full_list = loadtext(os.path.join(path_split,prefix+'.txt')) #train_list+val_list
full_list = loadtext(args.list)
# print(os.path.join(path_split,prefix+'.txt'))
print('num of scans:',len(full_list))
if(len(full_list)==0):
    raise RuntimeError('no scans')
#import sys
#sys.exit()
_, label_name_mapping, _ = getLabelMapping('nyu40',path_label_mapping)

stuff_list = ['blinds','ceiling','clothes','curtain','floor','otherprop', 'otherstructure', 'shower curtain', 'towel', 'wall']

path_annotation_base = os.path.join(pth_out, 'annotations')
annotation_path               = os.path.join(path_annotation_base, "instances_{}.json".format(prefix))
save_thing_path               = os.path.join(path_annotation_base, "entity_thing_{}.json".format(prefix))
save_stuff_path               = os.path.join(path_annotation_base, "entity_stuff_{}.json".format(prefix))
save_entity_path              = os.path.join(path_annotation_base, "entity_{}.json".format(prefix))
if not os.path.exists(path_annotation_base):
    os.makedirs(path_annotation_base)
'''create annotation file'''
def create_panoptic_file():
    x=dict()
    x['info'] = {
        'description': 'Entity dataset of 3RScan',
        'url': 'https://github.com/WaldJohannaU/3RScan',
        'version': '2.0',
        'year': 2019,
        'contributor': 'Wald, J., Wu, S.',
        'data_created': date.today().strftime('%Y-%m-%d'), 
        }
    x['licenses'] = [MIT_license]
    x['images'] = []
    x['annotations'] = []
    x['categories'] = []
    x['segment_info'] = []
    
    '''write category'''
    for idx, name in enumerate(nyu40_labels):
        x['categories'].append({
            'id': idx,
            'name': name
            })
    return x

#if not os.path.exists(annotation_path):
instance_annotations = create_panoptic_file()
#else:
#    instance_annotations = mmcv.load(annotation_path)


# annotation_file = dict()
# annotation_file[scan_id]= create_panoptic_file() #copy.deepcopy(instance_annotations)
# instance_annotations = annotation_file[scan_id]
counter=0
c_img_id=0
for scan_id in tqdm(full_list):
    path_info = os.path.join(path_data, scan_id, 'sequence','_info.txt')
    infos = loadtext(path_info)
    
    path_base_rgb = os.path.join(path_data, scan_id, 'sequence', name_format_img)
    path_base_inst = os.path.join(path_data, scan_id, 'sequence', name_format_inst)
    path_base_label = os.path.join(path_data, scan_id, 'sequence', name_format_label)
    path_semseg = os.path.join(path_data,scan_id,name_semseg)
    
    mapping = load_semseg(path_semseg,label_name_mapping)
    
    # save_base_path = os.path.join(pth_out,prefix,scan_id)
    save_base_path = os.path.join(pth_out,prefix)
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    
    # get num of frames
    num_frames = int(infos[-1].split('=')[-1].replace(' ', ''))
    for img_id in tqdm(range(num_frames), leave=False):
        img_path = path_base_rgb.format(img_id)
        
        '''process and export entity'''
        rgb = np.array(Image.open(img_path), dtype=np.uint8)
        panoptic_id              = np.array(Image.open(path_base_inst.format(img_id)), dtype=np.uint8)
        # labels              = np.array(Image.open(path_base_label.format(img_id)), dtype=np.uint8)
        width, height = panoptic_id.shape[-1],panoptic_id.shape[-2]
        
        panoptic_class_id     = np.zeros(panoptic_id.shape, dtype=np.uint8) + 255
        unique_ids = np.unique(panoptic_id)
        bounding_box = []
        # print('===============')
        for ii, entity_id in enumerate(unique_ids):
            if entity_id not in mapping: continue
            # if entity_id==0:continue
            label = mapping[entity_id]
            category = int(nyu40_labels.index(label))
            # print('{}: {}'.format(entity_id, mapping[entity_id]))

            '''write info'''
            panoptic_class_id[panoptic_id==entity_id]  = category
            mask     = (panoptic_id==entity_id).astype(np.uint8)
    
            finds_y, finds_x = np.where(mask==1)
            y1 = int(np.min(finds_y))
            y2 = int(np.max(finds_y))
            x1 = int(np.min(finds_x))
            x2 = int(np.max(finds_x))
            is_thing = int(label not in stuff_list) # true: is_thing 
            bounding_box.append([x1,y1,x2,y2,category,is_thing,entity_id])
        bounding_box = np.array(bounding_box)
        panoptic_info = np.stack((panoptic_id, panoptic_class_id), axis=0)
        anno_path = os.path.join(save_base_path, name_format_output.format(c_img_id) )
        np.savez(anno_path,map=panoptic_info, bounding_box=bounding_box)
        
        '''save image'''
        image_info = {
            'license': 0,
            'file_name': img_path,
            'height': height,
            'width': width,
            "id": c_img_id,
            'annotation_path': anno_path,
            }
        instance_annotations['images'].append(image_info)
        c_img_id += 1
    #break
    counter+=1
    # if counter > 1: break
        # print("{}, {}, {}".format(thread_idx, img_index, file_name))
        
mmcv.dump(instance_annotations, annotation_path)
