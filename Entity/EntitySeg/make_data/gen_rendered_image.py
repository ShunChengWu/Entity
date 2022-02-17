#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:33:42 2021

@author: sc
"""
# import os,io
# import zipfile
# import imageio
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# from PIL import Image
# import codeLib
# from codeLib.torch.visualization import show_tv_grid
# from codeLib.common import color_rgb, rand_24_bit,create_folder
# from codeLib.utils.util import read_txt_to_list
# from codeLib.object import BoundingBox
# from codeLib.utils.classification.labels import get_ScanNet_label_mapping#get_NYU40_color_palette, NYU40_Label_Names,get_ScanNet_label_mapping
# import torch
# import torchvision
# from torchvision.utils import draw_bounding_boxes
# from collections import defaultdict
# import json, glob, csv, sys,os, argparse
# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
from tqdm import tqdm
import argparse
import os
import subprocess, os, sys, time


helpmsg = 'Generate rendered image using the exe from 3RScan example'
parser = argparse.ArgumentParser(description=helpmsg,
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--scan3r_dir', default="/media/sc/space1/dataset/scannet/")
parser.add_argument('-exe','--exe',
                    default="/mnt/d/dataset/3RScan/c++/rio_renderer/build/rio_renderer_render_all", 
                    help="exe location")
parser.add_argument('-o','--outdir',default='sequence')
parser.add_argument('--thread', type=int, default=4, help='The number of threads to be used.')
args = parser.parse_args()

def process(scan_id):
    try:
        output = subprocess.check_output([args.exe, args.scan3r_dir, scan_id,args.outdir,'1'
                     ],
            stderr=subprocess.STDOUT)
        sys.stdout.write(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print('[Catched Error]', "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output.decode('utf-8'))) # omit errors

if __name__ =='__main__':
    folder =args.scan3r_dir
    scan_ids = os.listdir(folder)
    tmp=list()
    for s in scan_ids:
        full_path = os.path.join(folder,s)
        if os.path.isdir(full_path):
            tmp.append(s)
            # print(s)
    scan_ids = tmp
    n_workers=args.thread
    
    
    print('process {} scans with {} threads'.format(len(scan_ids), n_workers))
    if n_workers==0:
        for scan_id in tqdm(scan_ids):
            process(scan_id)
    else:
        process_map(process, scan_ids, max_workers=n_workers, chunksize=1 )