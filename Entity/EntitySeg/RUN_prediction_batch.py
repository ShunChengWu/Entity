import json


if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    # sys.path.append('../python/GCN')
# from utils.util import read_txt_to_list
# from utils import define
import subprocess, os, sys, time, json, math, argparse
import multiprocessing as mp
import numpy as np

parser = argparse.ArgumentParser(description='Run inference on 3RScan dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--basepath', type=str, 
                    default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan', help='', required=False)
parser.add_argument('--filename', '-f', type=str, 
                    default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/3RScan.json', help='', required=False)
parser.add_argument('--mode', '-m', type=str, 
                    default='train',choices=['train','val'], help='', required=False)
parser.add_argument('--config', '-cfg', type=str, 
                    default='configs/entity_r50_1x_3rscan_all_3.yaml', help='', required=False)
parser.add_argument('--output', '-o', type=str, 
                    default='/media/sc/SSD1TB2/prediction/', help='', required=False)
parser.add_argument('--checkpoint', '-ckpt', type=str, 
                    default='./output/r50_1x_3rscan_all_3/model_final.pth', help='', required=False)
# parser.add_argument('--filename', '-f', type=str, 
#                     default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/3RScan.json', help='', required=False)
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

if __name__ == '__main__':
    print(args)
    train_list, val_list = gen_splits(args.filename)
    print(len(train_list))
    print(len(val_list))
    if args.mode == 'train':
        flist = train_list
    elif args.mode == 'val':
        flist = val_list
    for scan_id in sorted(flist):
        pth_in = os.path.join(args.basepath,scan_id,'sequence/frame-*[0-9].color.jpg')
        pth_out = os.path.join(args.output,scan_id)
        print(pth_in)
        print(pth_out)
        # break
        process(pth_in,pth_out)
        # break
    pass