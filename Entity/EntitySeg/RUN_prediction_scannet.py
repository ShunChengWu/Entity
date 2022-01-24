import json


if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    # sys.path.append('../python/GCN')
# from utils.util import read_txt_to_list
# from utils import define
import h5py
import subprocess, os, sys, time, json, math, argparse
import multiprocessing as mp
import numpy as np
import imageio
from PIL import Image

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import copy

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup

from entityseg import *

from predictor import VisualizationDemo
import pdb

# constants
WINDOW_NAME = "Image Segmentation"



parser = argparse.ArgumentParser(description='Run inference on ScanNet dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--basepath', type=str, 
                    default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan', help='', required=False)
parser.add_argument('--filename', '-f', type=str, 
                    default='/media/sc/SSD1TB2/dataset/scannet/images.h5', help='', required=False)
parser.add_argument('--scan_id', '-s', type=str,default='scene0000_00', help='scan id. if empty, process all', 
                    required=False)
parser.add_argument('--split', '-t',type=str,
                    default='/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_train.txt',help='split file.')
parser.add_argument('--output', '-o', type=str, 
                    default='/media/sc/SSD1TB2/prediction/', help='', required=False)
parser.add_argument('--checkpoint', '-ckpt', type=str, 
                    default='./output/r50_1x_3rscan_all_3/model_final.pth', help='', required=False)
# parser.add_argument('--filename', '-f', type=str, 
#                     default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/3RScan.json', help='', required=False)

parser.add_argument(
        "--config",'-cfg',
        default="configs/entity_r50_1x_3rscan_all_3.yaml",
        metavar="FILE",
        help="path to config file",
    )

parser.add_argument(
       "--rotate",
       type=int,
       default=0,
       help="How many times the input image should be rotated (np.rot90). ",
   )
parser.add_argument(
    "--rotate-out",
    type=int,
    default=0,
    help="How many times the output image should be rotated (np.rot90). ",
)
    
parser.add_argument(
    "--input",
    nargs="+",
    help="A list of space separated input images; "
    "or a single glob pattern such as 'directory/*.jpg'",
)
# parser.add_argument(
#     "--output",
#     help="A file or directory to save output visualizations. "
#     "If not given, will show output in an OpenCV window.",
# )

parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.2,
    help="Minimum score for instance predictions to be shown",
)
   

parser.add_argument(
    "opts",
    help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
    "See config references at "
    "https://detectron2.readthedocs.io/modules/config.html#config-references",
    default=None,
    nargs=argparse.REMAINDER,
)
parser.add_argument("--save-npy", action="store_true", help="save output to npy")
parser.add_argument("--save-jpg", action="store_true", help="save output to jpg")

args = parser.parse_args()

exe_path='./demo_result_and_vis.py'

def process(pth_in, pth_out):
    startTime = time.time()
    try:
        output = subprocess.check_output(['python', exe_path, 
                                          '--input',pth_in,
                                          '--output',pth_out,
                                          '--config-file',args.config,
                                          '--rotate','3',
                                          '--rotate-out', '1',
                                          '--save-npy',
                                           '--save-jpg',
                                          'MODEL.WEIGHTS', args.checkpoint,
                                          'MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE', 'True',
                     ],
            stderr=subprocess.STDOUT)
        sys.stdout.write(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print('[Catched Error]', "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output.decode('utf-8'))) # omit errors
    endTime = time.time()
    return endTime-startTime

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def open_img_hdf5(path):
    return h5py.File(path,'r')

def gen_splits(pth_3rscan_json):
    with open(pth_3rscan_json,'r') as f:
        scan3r = json.load(f)
    
    train_list = list()
    val_list  = list()
    for scan in scan3r:
        ref_id = scan['reference']

        if scan['type'] == 'train':
            l = train_list
        elif scan['type'] == 'validation':
            l = val_list
        else:
            continue
        l.append(ref_id)
        for sscan in scan['scans']:
            l.append(sscan['reference'])
    
    return train_list ,val_list


def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors

def mask_to_boundary(mask, dilation_ratio=0.0008):
	"""
	Convert binary mask to boundary mask.
	:param mask (numpy array, uint8): binary mask
	:param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
	:return: boundary mask (numpy array)
	"""
	h, w = mask.shape
	img_diag = np.sqrt(h ** 2 + w ** 2)
	dilation = int(round(dilation_ratio * img_diag))
	if dilation < 1:
	    dilation = 1
	# Pad image so mask truncated by the image border is also considered as boundary.
	new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
	kernel = np.ones((3, 3), dtype=np.uint8)
	new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
	mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
	# G_d intersects G in the paper.
	return mask - mask_erode


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_entity_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

import sys

if __name__ == '__main__':
    print(args)
    # load H5
    data = open_img_hdf5(args.filename)
    
    read_txt_to_list(args.split)
    
    if args.split != '':
        flist = read_txt_to_list(args.split)
        print(len(flist))
    
    if args.scan_id != '':
        # check target scan exist
        if args.scan_id not in data:
            raise RuntimeError('cannot find target scan',args.scan_id)
        flist = [args.scan_id]
        print(len(flist))
        
    # init network
    mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    cfg = setup_cfg(args)
    
    demo = VisualizationDemo(cfg)
    colors = make_colors()
    
    for scan_id in sorted(flist):
        scan_data = data[scan_id]
        
        pth_out = os.path.join(args.output,scan_id)
        if not os.path.exists(pth_out):
            os.makedirs(pth_out)
        
        pbar = tqdm.tqdm(range(len(scan_data['rgb'])), disable=not args.output)
        for fid in pbar:
            img = scan_data['rgb'][fid]
            img = imageio.imread(img)
            img = Image.fromarray(img)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pbar.set_description("Processing %s" % fid)
            # use PIL, to be consistent with evaluation
            img = np.rot90(img,args.rotate)
            start_time = time.time()
            data = demo.run_on_image_wo_vis(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    fid,
                    "detected {} instances".format(len(data[0])),
                    time.time() - start_time,
                )
            )
            
            
            out_filename = os.path.join(pth_out, str(fid))
            
            ## save inference result, [0] original score by detection head, [1] mask rescoring score, [2] mask_id
            if args.save_npy:
                data[2] = np.rot90(data[2],args.rotate_out)
                
                ori_scores = data[0]
                scores = data[1]
                mask_id = data[2]
                np.savez(out_filename+".npz", ori_scores=ori_scores, scores=scores, mask_id=mask_id)

            ## save visualization
            if args.save_jpg:
                img = np.rot90(img,args.rotate_out)
                img_for_paste = copy.deepcopy(img)
                color_mask     = copy.deepcopy(img)
                masks_edge     = np.zeros(img.shape[:2], dtype=np.uint8)
                alpha  = 0.4
                count  = 0
                for index, score in enumerate(scores):
                    if score <= args.confidence_threshold:
                        break
                    color_mask[mask_id==count] = colors[count]
                    boundary = mask_to_boundary((mask_id==count).astype(np.uint8))
                    masks_edge[boundary>0] = 1
                    count += 1
                img_wm = cv2.addWeighted(img_for_paste, alpha, color_mask, 1-alpha, 0)
                img_wm[masks_edge==1] = 0
                fvis = np.concatenate((img, img_wm))
                cv2.imwrite(out_filename+".jpg",fvis)
            # break
        break
    pass