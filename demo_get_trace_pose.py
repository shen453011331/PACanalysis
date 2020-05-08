# this file is used to not only get locating of the contact points but also recontruct it in l-r point set
# and reconstruct horn points detected by Deep Learning method.
# then construct pantograph in 3D and get the pose

import argparse
import cv2
import time
import os
import pickle

from dataloader import *
from plotresult import *
from result_saving import *
from process3D import *
from analysis_data import *
from process import *
from locate import *
from tracker import *

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_pathL',  dest='data_pathL', type=str, default='F:/dataset/L', help='path of data')
parser.add_argument('--data_pathR',  dest='data_pathR', type=str, default='F:/dataset/R/JPEGImages', help='path of data')
parser.add_argument('--title',  dest='title', type=str, default='full_position_r', help='path of data')
parser.add_argument('--system_path',  dest='system_path', type=str, default='D:/clean/PACanalysis/', help='path of data')

parser.add_argument('--data_num',  dest='data_num', type=int, default=21732, help='number of data')
parser.add_argument('--meangrayr',  dest='meangrayr', type=float, default=90.0/255, help='line template 1')
parser.add_argument('--meangrayc',  dest='meangrayc', type=float, default=88.0/255, help='horn template 1')
parser.add_argument('--widpatchr',  dest='widpatchr', type=int, default=24, help='line template 2')
parser.add_argument('--widpatchr_r',  dest='widpatchr_r', type=int, default=17, help='line template 2 for right image')

parser.add_argument('--widpatchc',  dest='widpatchc', type=int, default=10, help='horn template 2')
parser.add_argument('--disr',  dest='disr', type=int, default=70, help='line template 3')
parser.add_argument('--disc',  dest='disc', type=int, default=64, help='horn template 3')
parser.add_argument('--thre',  dest='thre', type=int, default=25, help='max threshold for verify')
args = parser.parse_args()

locate_params = {'meangrayr': args.meangrayr,
                 'meangrayc': args.meangrayc,
                 'widpatchr': args.widpatchr,
                 'widpatchc': args.widpatchc,
                 'disr': args.disr,
                 'disc': args.disc}
locate_params_r = {'meangrayr': args.meangrayr,
                 'meangrayc': args.meangrayc,
                 'widpatchr': args.widpatchr_r,
                 'widpatchc': args.widpatchc,
                 'disr': args.disr,
                 'disc': args.disc}

if __name__ == '__main__':
    print('initilize parameters')
    # for left
    # init Loading Class
    ldata = DataLoader(args.data_pathL)
    left_horn_file = os.path.join(args.system_path, 'gt', 'hornDetect0820.txt')
    ldata.load_horn_files(left_horn_file)
    rdata = DataLoader(args.data_pathR)
    right_horn_file = os.path.join(args.system_path, 'gt', 'hornDetect0820r.txt')
    ldata.load_horn_files(right_horn_file)


    gt_file_l = os.path.join(args.system_path, 'gt')
    df_gt = ldata.load_gt_l(gt_file_l, 1000)# just for the first image
    gt_file_r = os.path.join(args.system_path, 'gt', 'R')
    df_gt_r = ldata.load_gt_l(gt_file_r, 1000)# just for the first image

    # init locate and track
    template = LocateTemplate(locate_params['widpatchr'], locate_params['widpatchc'],
                              locate_params['meangrayr'], locate_params['meangrayc'],
                              locate_params['disr'], locate_params['disc'])
    template_r = LocateTemplate(locate_params_r['widpatchr'], locate_params_r['widpatchc'],
                              locate_params_r['meangrayr'], locate_params_r['meangrayc'],
                              locate_params_r['disr'], locate_params_r['disc'])
    area = LocateArea([0, 400, 600, 1900], [300, 550, 800, 1600])
    locate = Locate(LocateParams(template, area, [1, -0.5], [-1, -3.5]))
    track = TrackerPAC(TrackParams(100, 50))
    locate_r = Locate(LocateParams(template_r, area, [3, -3], [1, 0.00001]))
    track_r = TrackerPAC(TrackParams(100, 50))
    # for right
    target_num = 181

    # init class for plot
    show = PlotResult()
    cv2.namedWindow('show_track', 0)
    cv2.resizeWindow("show_track", 1200, 500)
    cv2.moveWindow("show_track", 100, 100)

    # init output
    save_path = os.path.join(args.system_path, 'result', args.title)
    folder = os.path.exists(save_path)
    if not folder:
        os.mkdir(save_path)


    # init parameters
    sum_time, sum_time_r = 0, 0
    start_num = 0
    old_image, old_image_r = None, None
    b_re_track, b_re_track_r = False, False
    patch_l_tgt, patch_r_tgt = None, None

    print('all parameter inited finished')
    for i in range(start_num, args.data_num):
        img_idx = i + 1
        if img_idx % 100 == 0:
            print('image No. {} fps {}'.format(img_idx, fps))
        # load image
        image_l, fImage_l, _, _ = ldata.load(img_idx)
        image_r, fImage_r, _, _ = rdata.load_R(img_idx)
        if len(image_r[image_r < 1]) > 1e6:
            if img_idx % 100 == 0:
                print('image No. {} is black'.format(img_idx))
            pass
        else:
            old_image = image_l if old_image is None else old_image
            old_image_r = image_r if old_image_r is None else old_image_r

            # loading gt position for the first images and init the patch_tgt for hash similarity evalution
            points, b_re_track, fps, old_image, sum_time, track, locate = l_track(i, start_num, b_re_track, sum_time,
                                                                                  image_l, fImage_l, old_image,
                                                                                  df_gt, track, locate)
            points_r, b_re_track_r, fps_r, old_image_r,\
            sum_time_r, track_r, locate_r = l_track(i, start_num, b_re_track_r, sum_time_r,
                                                    image_r, fImage_r, old_image_r,
                                                    df_gt_r, track_r, locate_r)
            if img_idx == target_num:
                temp_df = df_gt_r[df_gt_r['number'] == img_idx]
                points = temp_df[['point_l_x', 'point_l_y', 'point_2_x', 'point_2_y']].values.reshape(2, 2)
                patch_l_tgt, patch_r_tgt = get_patch(image_r, points[0:1, :]), get_patch(image_r, points[1:, :])
                track_r.params.update_patch(patch_l_tgt, patch_r_tgt, 80)
            if img_idx == 17600:
                track.params.widthx = 200
                track_r.params.widthx = 200
                track.b_init = False
                track_r.b_init = False
            # contact points in two image has located successfully
            # to show,try to draw polar lines of the points


            # temp show in 2w sequence
            img = cv2.cvtColor(image_r, cv2.COLOR_GRAY2BGR)
            img2 = None
            if img_idx in locate_r.output_locate.index_list:
                img2 = show.draw_locate(img, locate_r.output_locate, img_idx)
            elif img_idx in track_r.output_track.index_list:
                img2 = show.draw_track(img, track_r.output_track, img_idx)
            else:
                print('no result for plot')
            if img2 is not None:
                cv2.imshow('show_track', img2)
                cv2.waitKey(1)

    cv2.destroyAllWindows()


    # save position_result
    # output = locate.output_locate
    # output_hal = open(os.path.join(save_path, 'test_locate.pkl'), 'wb')
    # str_chan = pickle.dumps(output)
    # output_hal.write(str_chan)
    # output_hal.close()
    # output = track.output_track
    # output_hal = open(os.path.join(save_path, 'test_track.pkl'), 'wb')
    # str_chan = pickle.dumps(output)
    # output_hal.write(str_chan)
    # output_hal.close()

    output = locate_r.output_locate
    output_hal = open(os.path.join(save_path, 'test_locate.pkl'), 'wb')
    str_chan = pickle.dumps(output)
    output_hal.write(str_chan)
    output_hal.close()
    output = track_r.output_track
    output_hal = open(os.path.join(save_path, 'test_track.pkl'), 'wb')
    str_chan = pickle.dumps(output)
    output_hal.write(str_chan)
    output_hal.close()
