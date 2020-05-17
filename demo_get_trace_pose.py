# this file is used to not only get locating of the contact points but also recontruct it in l-r point set
# and reconstruct horn points detected by Deep Learning method.
# then construct pantograph in 3D and get the pose

import argparse
import cv2
import time
import os
import pickle
from pyntcloud import PyntCloud
import open3d
import math

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
parser.add_argument('--title',  dest='title', type=str, default='full_position_lr', help='path of data')
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
    rdata.load_horn_files(right_horn_file)


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
    locate = Locate(LocateParams(template, area, [-1, -3.5], [1, -0.5]))
    track = TrackerPAC(TrackParams(100, 50))
    locate_r = Locate(LocateParams(template_r, area, [3, -3], [1, 0.00001]))
    track_r = TrackerPAC(TrackParams(100, 50))
    # for right
    target_num = 181

    # # init class for plot
    # show = PlotResult()
    # cv2.namedWindow('show_track', 0)
    # cv2.resizeWindow("show_track", 2400, 500)
    # cv2.moveWindow("show_track", 100, 100)

    #init for process 3D
    pro3d = Process3D()
    tgt_3d_horn = os.path.join(args.system_path, 'gt', 'mystd.ply')
    cloud = open3d.io.read_point_cloud(tgt_3d_horn)
    # open3d.visualization.draw_geometries([cloud], window_name="Open3D0")
    # cloud = PyntCloud.from_file(tgt_3d_horn)
    pan_points_3d = np.asarray(cloud.points)
    # x 列车行驶方向 [-16, 16] y 滑版方向[-700*, 700*] z垂直地面方向[23 ,-100]
    pan_points_3d[:, 1] = pan_points_3d[:, 1] - min(pan_points_3d[:, 1])
    pan_points_3d[:, 2] = pan_points_3d[:, 2] - min(pan_points_3d[:, 2])
    pan_length_tgt = max(pan_points_3d[:, 1]) - min(pan_points_3d[:, 1])
    pan_height_tgt = max(pan_points_3d[:, 2]) - min(pan_points_3d[:, 2])
    # x 列车行驶方向 [-16, 16] y 滑版方向[0, *] z垂直地面方向[0, *]



    # init output
    save_path = os.path.join(args.system_path, 'result', args.title)
    folder = os.path.exists(save_path)
    if not folder:
        os.mkdir(save_path)
    output_3d = OutPutResult()


    # init parameters
    sum_time, sum_time_r = 0, 0
    start_num = 0
    fps = 0
    old_image, old_image_r = None, None
    b_re_track, b_re_track_r = False, False
    patch_l_tgt, patch_r_tgt = None, None
    points_zero = np.zeros((2, 2), dtype=np.float32)
    b_load = 1

    # load results
    if b_load:
        with open(os.path.join(os.path.join(save_path, 'test_locate.pkl')), 'rb') as file:
            locate_output = pickle.loads(file.read())
        with open(os.path.join(os.path.join(save_path, 'test_locate_r.pkl')), 'rb') as file:
            locate_output_r = pickle.loads(file.read())
        with open(os.path.join(os.path.join(save_path, 'test_track_r.pkl')), 'rb') as file:
            track_output_r = pickle.loads(file.read())
        with open(os.path.join(os.path.join(save_path, 'test_track.pkl')), 'rb') as file:
            track_output = pickle.loads(file.read())


    print('all parameter inited finished')
    for i in range(start_num, 20000):  # args.data_num):
        img_idx = i + 1
        if img_idx % 100 == 0:
            print('image No. {} fps {}'.format(img_idx, fps))
        if not b_load:
            # load image
            image_l, fImage_l, _, _ = ldata.load(img_idx)
            image_r, fImage_r, _, _ = rdata.load_R(img_idx)
            if len(image_r[image_r < 1]) > 1e6:
                if img_idx % 100 == 0:
                    print('image No. {} is black'.format(img_idx))
                points, points_r = points_zero, points_zero
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
            # get 3d contact points
            points_3d = pro3d.reconstruct3D(np.mean(points, 0), np.mean(points_r, 0))
        else:
            # load points by files

            dict_data = None
            if img_idx in locate_output.index_list:
                index = locate_output.index_list.index(img_idx)
                dict_data = locate_output.result_list[index]
            elif img_idx in track_output.index_list:
                index = track_output.index_list.index(img_idx)
                dict_data = track_output.result_list[index]
            points = np.vstack((dict_data['points_l'], dict_data['points_r'])) \
                if dict_data is not None else points_zero

            dict_data_r = None
            if img_idx in locate_output_r.index_list:
                index = locate_output_r.index_list.index(img_idx)
                dict_data_r = locate_output_r.result_list[index]
            elif img_idx in track_output_r.index_list:
                index = track_output_r.index_list.index(img_idx)
                dict_data_r = track_output_r.result_list[index]
            points_r = np.vstack((dict_data_r['points_l'], dict_data_r['points_r'])) \
                if dict_data_r is not None else points_zero
            points_3d = pro3d.reconstruct3D(np.mean(points, 0), np.mean(points_r, 0))
        # load 羊角
        lhorn_l, lhorn_r = ldata.load_hornPoints_from_df(ldata.df, i + 1)
        rhorn_l, rhorn_r = rdata.load_hornPoints_from_df(rdata.df, i + 1)
        p3d_lhorn = pro3d.reconstruct3D(lhorn_l, rhorn_l)
        p3d_rhorn = pro3d.reconstruct3D(lhorn_r, rhorn_r)
        # x 滑版方向 y 列车行驶方向  z 垂直地面方向
        # load 羊角三维文件
        # 根据三点获取当前滑版位姿，主要包括抬升量和倾斜度两个指标，以
        # 及接触点到滑版的距离和两个羊角间距离作为一个缩放量
        distance, height, length, theta = pro3d.calculateShift(points_3d, p3d_lhorn, p3d_rhorn)
        output_3d.save_results(points_3d, p3d_lhorn, p3d_rhorn, height, length, distance, theta, img_idx)
        # distance is 接触点到中心的偏移量，height 是滑版的空间高度， length，滑版空间长度
        # rotate_matrix = np.array([[math.cos(theta), math.sin(theta)],
        #                          [-math.sin(theta), math.cos(theta)]])
        # temp_pan_3d = pan_points_3d.copy()
        # # zoom
        # temp_pan_3d[:, 1] = temp_pan_3d[:, 1] * length / pan_length_tgt
        # temp_pan_3d[:, 2] = temp_pan_3d[:, 2] * height / pan_height_tgt
        # # rotate
        # temp_pan_3d[:, 1:] = np.dot(rotate_matrix, pan_points_3d[:, 1:].T).T
        # # shift
        # temp_pan_3d[:, 0] = temp_pan_3d[:, 0] + np.squeeze(p3d_lhorn)[1]
        # temp_pan_3d[:, 1] = temp_pan_3d[:, 1] + np.squeeze(p3d_lhorn)[0]
        # temp_pan_3d[:, 2] = temp_pan_3d[:, 2] + np.squeeze(p3d_lhorn)[2]



    #         # contact points in two image has located successfully
    #         # to show,try to draw polar lines of the points
    #         lines, k = pro3d.getPolarLine(np.mean(points, 0), np.mean(points_r, 0))
    #         line_b, line_k = -lines[2] / lines[1], -lines[0] / lines[1]
    #         # plot l
    #         img = cv2.cvtColor(image_l, cv2.COLOR_GRAY2BGR)
    #         if img_idx in locate.output_locate.index_list:
    #             img = show.draw_locate(img, locate.output_locate, img_idx)
    #         elif img_idx in track.output_track.index_list:
    #             img = show.draw_track(img, track.output_track, img_idx)
    #         # plot r
    #         img_r = cv2.cvtColor(image_r, cv2.COLOR_GRAY2BGR)
    #         if img_idx in locate_r.output_locate.index_list:
    #             img_r = show.draw_locate(img_r, locate_r.output_locate, img_idx)
    #         elif img_idx in track_r.output_track.index_list:
    #             img_r = show.draw_track(img_r, track_r.output_track, img_idx)
    #         img_r = show.draw_line_on_image(img_r, line_k, line_b)
    #         stack_image = np.hstack((img, img_r))
    #         cv2.imshow('show_track', stack_image)
    #         cv2.waitKey(1)
    #
    # cv2.destroyAllWindows()
    output = output_3d
    output_hal = open(os.path.join(save_path, 'result_3d.pkl'), 'wb')
    str_chan = pickle.dumps(output)
    output_hal.write(str_chan)
    output_hal.close()

    if not b_load:
        # save position_result
        output = locate.output_locate
        output_hal = open(os.path.join(save_path, 'test_locate.pkl'), 'wb')
        str_chan = pickle.dumps(output)
        output_hal.write(str_chan)
        output_hal.close()
        output = track.output_track
        output_hal = open(os.path.join(save_path, 'test_track.pkl'), 'wb')
        str_chan = pickle.dumps(output)
        output_hal.write(str_chan)
        output_hal.close()

        output = locate_r.output_locate
        output_hal = open(os.path.join(save_path, 'test_locate_r.pkl'), 'wb')
        str_chan = pickle.dumps(output)
        output_hal.write(str_chan)
        output_hal.close()
        output = track_r.output_track
        output_hal = open(os.path.join(save_path, 'test_track_r.pkl'), 'wb')
        str_chan = pickle.dumps(output)
        output_hal.write(str_chan)
        output_hal.close()
