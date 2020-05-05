# using for test different choice of tracking and verify the effeciency of tracking
# 5.2 21:59
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
from tracker import *
import pickle
parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_pathL',  dest='data_pathL', type=str, default='F:/dataset/L', help='path of data')
parser.add_argument('--data_pathR',  dest='data_pathR', type=str, default='F:/dataset/R/JPEGImages', help='path of data')

parser.add_argument('--data_num',  dest='data_num', type=int, default=21732, help='number of data')
parser.add_argument('--meangrayr',  dest='meangrayr', type=float, default=90.0/255, help='line template 1')
parser.add_argument('--meangrayc',  dest='meangrayc', type=float, default=88.0/255, help='horn template 1')
parser.add_argument('--widpatchr',  dest='widpatchr', type=int, default=24, help='line template 2')
parser.add_argument('--widpatchc',  dest='widpatchc', type=int, default=10, help='horn template 2')
parser.add_argument('--disr',  dest='disr', type=int, default=70, help='line template 3')
parser.add_argument('--disc',  dest='disc', type=int, default=64, help='horn template 3')
parser.add_argument('--thre',  dest='thre', type=int, default=25, help='max threshold for verify')
args = parser.parse_args()
ldata = DataLoader(args.data_pathL)
locate_params = {'meangrayr': args.meangrayr,
                 'meangrayc': args.meangrayc,
                 'widpatchr': args.widpatchr,
                 'widpatchc': args.widpatchc,
                 'disr': args.disr,
                 'disc': args.disc}
track_params = {}
refine_params = {}
verify_params = {'thre': args.thre}
lprocess = ContactLocate(locate_params, track_params, refine_params, verify_params)
testShow = PlotResult()
system_path = 'D:/clean/PACanalysis/'
saveResult = SaveResult(system_path)
pro3d = Process3D()
ana_data = AnalysisData(saveResult, pro3d, testShow)

if __name__ == '__main__':
    template = LocateTemplate(locate_params['widpatchr'], locate_params['widpatchc'],
                              locate_params['meangrayr'], locate_params['meangrayc'],
                              locate_params['disr'], locate_params['disc'])
    area = LocateArea([0, 400, 600, 1900], [300, 550, 800, 1600])
    locate = Locate(LocateParams(template, area))
    track = TrackerPAC(TrackParams(100, 50))

    num = 1000 # 1000*n
    gt_file_l = 'D:\PACanalysis\gt'
    df_gt = ldata.load_gt_l(gt_file_l, num)

    title = 'track_refine_hash'
    with open(os.path.join(os.path.join(system_path, 'result', title, 'test_track.pkl')), 'rb') as file:
        track_output = pickle.loads(file.read())
    with open(os.path.join(os.path.join(system_path, 'result', title, 'test_locate.pkl')), 'rb') as file:
        locate_output = pickle.loads(file.read())
    output_path = os.path.join(system_path, 'output', title)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    idx_list = pd.read_csv('track_temp2.csv')['number'].values.tolist()
    for idx in idx_list:
        if idx % 100 == 0:
            print('image No. {}'.format(idx))
        if idx % 1 == 0:
            output_name = os.path.join(output_path, '{:06d}.png'.format(idx))
            output_name2 = os.path.join(output_path, '{:06d}_locate.png'.format(idx))
            filepath = os.path.join('F:/dataset/L', '{:06d}.bmp'.format(idx))
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if idx in locate_output.index_list:
                img2 = testShow.draw_locate(img, locate_output, idx)
                cv2.imwrite(output_name2, img2)
            if idx in track_output.index_list:
                img = testShow.draw_track(img, track_output, idx)
                cv2.imwrite(output_name, img)
