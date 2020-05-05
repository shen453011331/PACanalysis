# using for test different choice of tracking and verify the effeciency of tracking
# 5.2 21:59
import argparse
import cv2
import time
import os
from dataloader import *
# from contactlocate import *
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
parser.add_argument('--title',  dest='title', type=str, default='track_refine_hash_full_l', help='path of data')

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
# lprocess = ContactLocate(locate_params, track_params, refine_params, verify_params)
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

    num = 27100  # 1000*n
    gt_file_l = 'D:\PACanalysis\gt'
    df_gt = ldata.load_gt_l(gt_file_l, 1000)

    title = args.title
    # track_refine_hash_modify is modify paramter update with hash 20, speed up hash, and simulate when track error
    save_path = '%sresult/%s' % (system_path, title)
    folder = os.path.exists(save_path)
    if not folder:
        os.mkdir(save_path)

    sumtime = 0
    fps = 10
    start_num = 0
    old_image = None
    b_re_track = False
    patch_l_tgt, patch_r_tgt = None, None
    for i in range(start_num, num):
        img_idx = i + 1
        if img_idx % 100 == 0:
            print('image No. {} fps {}'.format(img_idx, fps))
        # load image
        image_l, fImage_l, _, _ = ldata.load(img_idx)
        old_image = image_l if old_image is None else old_image
        # loading gt position for the first images
        # and init the patch_tgt for hash similarity evalution
        if i == start_num:
            temp_df = df_gt[df_gt['number'] == img_idx]
            points = temp_df[['point_l_x', 'point_l_y', 'point_2_x', 'point_2_y']].values.reshape(2, 2)
            # track refine
            points, ok, elapsed_t = track.do_track_refine(img_idx, image_l, image_l, points, b_re_init=True)
            # track refine hash
            # points, ok, elapsed_t = track.do_track_refine_hash(img_idx, image_l, image_l, points, b_re_init=True)
            patch_l, patch_r = get_patch(image_l, points[0:1, :]), get_patch(image_l, points[1:, :])
            patch_l_tgt, patch_r_tgt = patch_l, patch_r
            track.params.update_patch(patch_l_tgt, patch_r_tgt)
        # for the tracking failed, re-locate  by locating method
        elif b_re_track:
            print('locate in NO.{}'.format(img_idx))
            # after testing in demo_test_locate, we choose locate + refine + update

            points, elapsed = locate.do_locate_refine(fImage_l, img_idx, 1)
            # update can be modifyed next by choosing a fixed hash value, for example < 20
            if verify_locate(locate.output_locate, img_idx):
                # if locate success, track can be inited
                # only track
                # points, ok, elapsed_t = track.do_track(img_idx, image_l, image_l, points, b_re_init=True)
                # track refine
                # points, ok, elapsed_t = track.do_track_refine(img_idx, image_l, image_l, points, b_re_init=True)
                # track refine hash
                points, ok, elapsed_t = track.do_track_refine_hash(img_idx, image_l, image_l, points, b_re_init=True)
                if ok and max(track.hash) < 20:
                    locate.do_update(fImage_l, img_idx)
                    print('NO. {} update parameters ,hash {},{}'.format(img_idx, track.hash[0], track.hash[1]))
                    # locate.do_update(fImage_l, img_idx)
                sumtime = sumtime + elapsed + elapsed_t
                if ok:
                    b_re_track = False
                    print('track init success in NO.{}'.format(img_idx))
                else:
                    b_re_track = True
                    print('track init error in NO.{}'.format(img_idx))
                # b_re_track = False if ok else True
            else:
                b_re_track = True
                print('locate verify error in NO.{}'.format(img_idx))
        else:
            # if last frame track success, continue tracking
            # only track
            # points, ok, elapsed_t = track.do_track(img_idx, old_image, image_l)
            # track refine
            # points, ok, elapsed_t = track.do_track_refine(img_idx, old_image, image_l)
            # track refine hash
            points, ok, elapsed_t = track.do_track_refine_hash(img_idx, old_image, image_l)

            sumtime = sumtime + elapsed_t
            if ok:
                b_re_track = False

                # print('track  success')
            else:
                b_re_track = True
                points = track.get_simulate_points(img_idx)
                print('track  error in NO.{}'.format(img_idx))
            # b_re_track = False if ok else True
        fps = 1. / (sumtime / (img_idx-start_num)) if sumtime !=0 else fps
        # this is using to plot result for the error between gt
        # ana_data.compare_with_gt(df_gt, points, img_idx)

    output = locate.output_locate
    output_hal = open(os.path.join(system_path, 'result', title, 'test_locate.pkl'), 'wb')
    str_chan = pickle.dumps(output)
    output_hal.write(str_chan)
    output_hal.close()
    output = track.output_track
    output_hal = open(os.path.join(system_path, 'result', title, 'test_track.pkl'), 'wb')
    str_chan = pickle.dumps(output)
    output_hal.write(str_chan)
    output_hal.close()


    # x_data = ana_data.df_gt_err['number'].values
    # x_data = x_data.tolist()
    # y_data = ana_data.df_gt_err['l_err'].values
    # y_data = y_data.tolist()
    # #  plot dist1
    # testShow.plot_lines(x_data, y_data, '%s/gt_l_err_%s.html' % (save_path, title), 'gt_l_err')
    # y_data = ana_data.df_gt_err['r_err'].values
    # y_data = y_data.tolist()
    # #  plot dist1
    # testShow.plot_lines(x_data, y_data, '%s/gt_r_err_%s.html' % (save_path, title), 'gt_r_err')