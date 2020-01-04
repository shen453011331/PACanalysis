# locate points and horn in both two image
# adding mutiple analysis of the results
# 2019.1.2 16:10
import argparse
import cv2
import time
# import torch
# from torch.autograd import Variable
# import torch.utils.data as data
# from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
#     COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
# import torchvision

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
parser.add_argument('--meangrayr',  dest='meangrayr', type=float, default=190.0/255, help='line template 1')
parser.add_argument('--meangrayc',  dest='meangrayc', type=float, default=88.0/255, help='horn template 1')
parser.add_argument('--widpatchr',  dest='widpatchr', type=int, default=18, help='line template 2')
parser.add_argument('--widpatchc',  dest='widpatchc', type=int, default=10, help='horn template 2')
parser.add_argument('--disr',  dest='disr', type=int, default=160, help='line template 3')
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
rprocess = ContactLocate(locate_params, track_params, refine_params, verify_params)
testShow = PlotResult()
system_path = 'C:/Users/Administrator/Documents/sy_paper_contactAnalysis/new/'
saveResult = SaveResult(system_path)
pro3d = Process3D()
ana_data = AnalysisData(saveResult, pro3d, testShow)

if __name__ == '__main__':
    sumTime_L = 0
    sumTime_R = 0
    lprocess.show = testShow
    rprocess.show = testShow
    left_horn_file = system_path + 'gt/hornDetect0820.txt'
    right_horn_file = system_path + 'gt/hornDetect0820r.txt'
    ldata.load_horn_files(left_horn_file)
    rdata.load_horn_files(right_horn_file)

    for i in range(5000):
        print('# %d:'% (i+1))
        image_l, fImage_l, _, _ = ldata.load(i + 1)
        image_r, fImage_r, _, _ = rdata.load_R(i + 1)
        # testShow.plot_two_images(image_l, image_r)
        points_lft, points_l_lft, time_process_l = lprocess.do_multiTrack_update(image_l, fImage_l, i + 1)
        points_rgt, points_l_rgt, time_process_r = rprocess.do_multiTrack_update(image_r, fImage_r, i + 1)
        sumTime_L = sumTime_L + time_process_l
        fps_L = (i + 1) / sumTime_L
        sumTime_R = sumTime_R + time_process_r
        fps_R = (i + 1) / sumTime_R
        # load 羊角
        lhorn_l, lhorn_r = ldata.load_hornPoints_from_df(ldata.df, i + 1)
        rhorn_l, rhorn_r = rdata.load_hornPoints_from_df(rdata.df, i + 1)
        # get 3D points
        P = pro3d.reconstruct3D(np.mean(points_lft, 0), np.mean(points_rgt, 0))
        P = np.array([[0, 0, 0]]).T if P is None else P
        P_lhorn = pro3d.reconstruct3D(lhorn_l, rhorn_l)
        P_rhorn = pro3d.reconstruct3D(lhorn_r, rhorn_r)
        P_lhorn = np.array([[0, 0, 0]]).T if P_lhorn is None else P_lhorn
        P_rhorn = np.array([[0, 0, 0]]).T if P_rhorn is None else P_rhorn
        # distance, height = pro3d.calculateShift(P, P_lhorn, P_rhorn)
        # pro3d.getPolarLine(lhorn_l[0],lhorn_l[1],rhorn_l[0],rhorn_l[1])
        # dist = np.linalg.norm(P_lhorn-P_rhorn)
        # print('distance between two horn is %f' % dist)
        result_value_file = 'result_value.csv'

        saveResult.save_all_value(i+1, points_l_lft, points_l_rgt, points_lft, points_rgt,
                                  lhorn_l, lhorn_r, rhorn_l, rhorn_r,
                                  P, P_lhorn, P_rhorn, result_value_file)

        ana_data.analysis_all(saveResult.df_all_value, i+1)

