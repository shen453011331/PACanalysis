# this file can load locate output files and output image with lines and circles with fix index
import argparse
import cv2
import time
import os
from dataloader import *
from contactlocate import *
from plotresult import *
from result_saving import *
from process3D import *
from analysis_data import *
from locate import *
import pickle
parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_pathL',  dest='data_pathL', type=str, default='F:/dataset/L', help='path of data')
args = parser.parse_args()

ldata = DataLoader(args.data_pathL)

if __name__ == '__main__':
    idx_list = [i  for i in range(9900, 10000)]
    # idx_list = [i+1 for i in range(20)]
    system_path = 'D:/clean/PACanalysis/'

    testShow = PlotResult()
    gt_file_l = 'D:\PACanalysis\gt'
    df_gt = ldata.load_gt_l(gt_file_l, 5000)
    title = 'locate_refine_update'
    with open(os.path.join(os.path.join(system_path, 'result', title, 'test_locate.pkl')), 'rb') as file:
        locate_output = pickle.loads(file.read())
    output_path = os.path.join(system_path, 'output', title)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for idx in idx_list:
        output_name = os.path.join(output_path, '{:06d}.png'.format(idx))
        filepath = os.path.join('F:/dataset/L', '{:06d}.bmp'.format(idx))
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = testShow.draw_locate(img, locate_output, idx)
        cv2.imwrite(output_name, img)