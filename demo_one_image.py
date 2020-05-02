# 2020.1.22 for only detect one image and test the value in different mode

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
    print('init success')
    sumTime_L = 0
    lprocess.show = testShow
    lprocess.save = saveResult
    lprocess.ana = ana_data
    left_horn_file = system_path + 'gt/hornDetect0820.txt'
    ldata.load_horn_files(left_horn_file)
    gt_file_l = 'D:\PACanalysis\gt'
    title = 'test'
    save_path = '%sresult/%s' % (system_path, title)
    folder = os.path.exists(save_path)
    if not folder:
        os.mkdir(save_path)

    df_gt = ldata.load_gt_l(gt_file_l, 5000)
    thre_1 = [17, 22]
    thre_2 = 6
    thre_2_2 = 6
    startNum = 0
    lprocess.startNum = startNum+1
    for i in range(startNum, 1000):

        img_idx = i + 1
        if img_idx % 10 == 0:
            print('image No. %d' % img_idx)
        image_l, fImage_l, _, _ = ldata.load(img_idx)
        points_r, _ = lprocess.corner_locating_seprete(image_l, fImage_l, img_idx)

        ana_data.compare_with_gt(df_gt, points_r, img_idx)
        # saveing in lprocess.do_analysis
        # print(points_r)
        # lprocess.show.plot_locate_result(image_l, points_r, 0)

    # saving and analysis after processing
    print('locate number: %d' % lprocess.locate_num)

    err1, err2 = ana_data.get_gt_error()
    file = open('%s/%s.txt'% (save_path, title), 'w')
    file.write('err1:%f,err2:%f, fps:%f' % (err1, err2, lprocess.fps))
    print(err1, err2)
    print('fps: %0.2f' % lprocess.fps)
    x_data = ana_data.df_gt_err['number'].values
    x_data = x_data.tolist()
    y_data = ana_data.df_gt_err['l_err'].values
    y_data = y_data.tolist()
    #  plot dist1
    testShow.plot_lines(x_data, y_data, '%s/gt_l_err_%s.html' % (save_path, title), 'gt_l_err')
    y_data = ana_data.df_gt_err['r_err'].values
    y_data = y_data.tolist()
    #  plot dist1
    testShow.plot_lines(x_data, y_data, '%s/gt_r_err_%s.html' % (save_path, title), 'gt_r_err')
    saveResult.df_contact.to_csv('%s/%s.csv' % (save_path, title), columns=saveResult.contact_volumn)




