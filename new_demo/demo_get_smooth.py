# 2020.7.30
# this file is used to output smooth with input frame sequence
# input file_path of the sequence
# output .csv which is the smooth value of each images
import argparse

from init_param import *
from process3D import *
from analysis import *

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_pathL',  dest='data_pathL', type=str, default='F:/dataset/L', help='path of data')
parser.add_argument('--data_pathR',  dest='data_pathR', type=str, default='F:/dataset/R/JPEGImages', help='path of data')
parser.add_argument('--system_path',  dest='system_path', type=str, default='D:/clean/PACanalysis/', help='path of data')

args = parser.parse_args()


if __name__ == '__main__':
    ldata, locate, track, df_gt, \
    rdata, locate_r, track_r, df_gt_r = do_init(args.data_pathL, args.data_pathR, args.system_path)
    pro3d = Process3D()
    ana = SmoothAnalysis()

    sum_time, sum_time_r = 0, 0
    start_num = 0
    fps = 0
    old_image, old_image_r = None, None
    b_re_track, b_re_track_r = False, False
    patch_l_tgt, patch_r_tgt = None, None
    points_zero = np.zeros((2, 2), dtype=np.float32)

    for i in range(start_num, 100):
        img_idx = i + 1
        image_l, fImage_l, _, _ = ldata.load(img_idx)
        image_r, fImage_r, _, _ = rdata.load_R(img_idx)
        if len(image_r[image_r < 1]) > 1e6:
            points, points_r = points_zero, points_zero
        else:
            points, b_re_track, fps, old_image, \
            sum_time, track, locate = locate_contact(i, start_num, b_re_track, sum_time,
                                                     image_l, fImage_l, old_image,
                                                     df_gt, track, locate)
            points_r, b_re_track_r, fps_r, old_image_r, \
            sum_time_r, track_r, locate_r = locate_contact(i, start_num, b_re_track_r, sum_time_r,
                                                           image_r, fImage_r, old_image_r,
                                                           df_gt_r, track_r, locate_r)
        points_3d = pro3d.reconstruct3D(np.mean(points, 0), np.mean(points_r, 0))
        lhorn_l, lhorn_r = ldata.load_hornPoints_from_df(ldata.df, i + 1)
        rhorn_l, rhorn_r = rdata.load_hornPoints_from_df(rdata.df, i + 1)
        p3d_lhorn = pro3d.reconstruct3D(lhorn_l, rhorn_l)
        p3d_rhorn = pro3d.reconstruct3D(lhorn_r, rhorn_r)
        # x 滑版方向 y 列车行驶方向  z 垂直地面方向
        # load 羊角三维文件
        # 根据三点获取当前滑版位姿，主要包括抬升量和倾斜度两个指标，以
        # 及接触点到滑版的距离和两个羊角间距离作为一个缩放量
        distance, height, length, theta = pro3d.calculateShift(points_3d, p3d_lhorn, p3d_rhorn)
        ana.get_smooth_para_df(img_idx, ldata, rdata,
                               p3d_lhorn, p3d_rhorn, points_3d,
                               points, points_r,
                               lhorn_l, lhorn_r, rhorn_l, rhorn_r,
                               distance, theta, height)
        p_smooth, p_lachu, [p_theta, p_h], \
        [p_c_1, p_c_2, p_c_3, p_c_4], \
        [p_c_1_y, p_c_2_y, p_c_3_y, p_c_4_y] = ana.get_smoothness_value(img_idx)
        print(p_smooth)



