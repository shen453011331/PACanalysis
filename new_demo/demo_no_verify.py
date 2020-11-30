# 2020.8.29
# this file is to save results for the method with no update
# for now, save results of left image first
import argparse

from init_param import *
from process3D import *
from analysis import *
from plotresult import *
from output import *
import datetime

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
    show = PlotResult()
    result = Result()

    sum_time, sum_time_r = 0, 0
    start_num = 0
    fps = 0
    old_image, old_image_r = None, None
    b_re_track, b_re_track_r = False, False
    patch_l_tgt, patch_r_tgt = None, None
    points_zero = np.zeros((2, 2), dtype=np.float32)
    class Result_test(object):
        def __init__(self):
            self.hash_result = np.zeros((21732, 2), dtype=np.float32)
            self.locate_result = np.zeros((21732, 1), dtype=np.float32)
            self.points_result = np.zeros((21732, 4), dtype=np.float32)
            self.shift_result = np.zeros((21732, 2), dtype=np.float32)
            self.tem_result = np.zeros((21732, 4), dtype=np.float32)
    result_test = Result_test()
    for i in range(start_num, 21732):
        img_idx = i + 1
        image_l, fImage_l, _, _ = ldata.load(img_idx)
        start = time.clock()
        # image_r, fImage_r, _, _ = rdata.load_R(img_idx)
        if len(image_l[image_l < 1]) > 1e6:
            points, points_r = points_zero, points_zero
        else:
            result_test.locate_result[i, 0] = 1 if b_re_track_r else 0
            points, b_re_track, fps, old_image, \
            sum_time, track, locate = locate_contact_no_update(i, start_num, b_re_track, sum_time,
                                                     image_l, fImage_l, old_image,
                                                     df_gt, track, locate)
            # points_r, b_re_track_r, fps_r, old_image_r, \
            # sum_time_r, track_r, locate_r = locate_contact(i, start_num, b_re_track_r, sum_time_r,
            #                                                image_r, fImage_r, old_image_r,
            #                                                df_gt_r, track_r, locate_r)
        roi = track.getRect(points)
        result.update(roi, time.clock()-start)
        result_test.points_result[i, :] = points.reshape([1, 4])
        result_test.tem_result[i, :] = [locate.params.template.mean_gray_r, locate.params.template.mean_gray_c,
                                        locate.params.template.dis_r, locate.params.template.dis_c]
        if len(track.hash) ==2:
            result_test.hash_result[i, 0] = track.hash[0]
            result_test.hash_result[i, 1] = track.hash[1]
        if len(track.shift) ==2:
            result_test.shift_result[i, 0] = track.shift[0]
            result_test.shift_result[i, 1] = track.shift[1]
        if (i + 1) % 1000 == 0:
            print('{} image has processed'.format(i))
            print('current time is {}'.format(datetime.datetime.now()))
            save_result(result, os.path.join(os.path.curdir, 'result',
                                             '{}-base-{}-no_update.pkl'.format(track.tracker.__class__.__name__[7:],
                                                                       i + 1)))
            save_result(result_test, os.path.join(os.path.curdir, 'result', 'output_test_no_update.pkl'))
    save_result(result, os.path.join(os.path.curdir, 'result',
                                     '{}-base-no_update.pkl'.format(track.tracker.__class__.__name__[7:])))
