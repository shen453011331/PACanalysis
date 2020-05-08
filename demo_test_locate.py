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
rdata = DataLoader(args.data_pathR)

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

# def show_locate(locate_output, idx, b_adding_points=1):
#     filepath = os.path.join('F:/dataset/L', '{:06d}.bmp'.format(idx))
#     img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     if b_adding_points:
#         img = testShow.circle_points_on_image(img, locate_output.)




if __name__ == '__main__':
    template = LocateTemplate(locate_params['widpatchr'], locate_params['widpatchc'],
                              locate_params['meangrayr'], locate_params['meangrayc'],
                              locate_params['disr'], locate_params['disc'])
    area = LocateArea([0, 400, 600, 1900], [300, 550, 800, 1600])
    locate = Locate(LocateParams(template, area))
    num = 5000 # 1000*n
    gt_file_l = 'D:\PACanalysis\gt'
    df_gt = ldata.load_gt_l(gt_file_l, num)
    title = 'locate_r'
    save_path = '%sresult/%s' % (system_path, title)
    folder = os.path.exists(save_path)
    if not folder:
        os.mkdir(save_path)
    sumtime = 0
    fps = 10
    for i in range(num):
        img_idx = i + 1
        if img_idx % 100 == 0:
            print('image No. {} fps {}'.format(img_idx, fps))
        # image_l, fImage_l, _, _ = ldata.load(img_idx)
        image_l, fImage_l, _, _ = rdata.load_R(img_idx)
        # only locate
        # points, elapsed = locate.do_locate(fImage_l, img_idx)
        # locate + refine
        # points, elapsed = locate.do_locate_refine(fImage_l, img_idx)
        # update
        points, elapsed = locate.do_locate_refine(fImage_l, img_idx, 1)
        locate.do_update(fImage_l, img_idx)
        sumtime = sumtime + elapsed
        fps = 1./(sumtime/img_idx)
        ana_data.compare_with_gt(df_gt, points, img_idx)
    output = locate.output_locate
    output_hal = open(os.path.join(system_path, 'result', title, 'test_locate.pkl'), 'wb')
    str_chan = pickle.dumps(output)
    output_hal.write(str_chan)
    output_hal.close()
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
