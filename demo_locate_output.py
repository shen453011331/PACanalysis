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



if __name__ == '__main__':
    testShow = PlotResult()
    system_path = 'D:/clean/PACanalysis/'
    saveResult = SaveResult(system_path)
    pro3d = Process3D()
    ana_data = AnalysisData(saveResult, pro3d, testShow)

    gt_file_l = 'D:\PACanalysis\gt'
    num = 10000
    df_gt = ldata.load_gt_l(gt_file_l, num)
    title = 'locate_refine_update'

    with open(os.path.join(os.path.join(system_path, 'result', title, 'test_locate.pkl')), 'rb') as file:
        locate_output = pickle.loads(file.read())
    err_locate_num = 0

    for i in range(num):
        img_idx = i + 1
        if img_idx % 100 == 0:
            print('image No. {}'.format(img_idx))
        dict_data = locate_output.result_list[img_idx - 1]
        if not verify_locate(locate_output, img_idx):
            err_locate_num = err_locate_num + 1
        else:
            points = np.vstack((dict_data['points_l'], dict_data['points_r']))
            ana_data.compare_with_gt(df_gt, points, img_idx)


    print('locate not success {}'.format(1-len(ana_data.df_gt_err)/num))
    df_gt_err = ana_data.df_gt_err
    df_temp = df_gt_err[(df_gt_err['l_err'] < 10) & (df_gt_err['r_err'] < 10)]
    print('locate missing {}, other{}'.format((len(df_gt_err) - len(df_temp))/num, len(df_temp)/num))
    accuracy = (np.mean(df_temp['l_err'].values) + np.mean(df_temp['r_err'].values))/2
    print('locate precise is {:.2f}'.format(accuracy))
