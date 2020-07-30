import os
from dataloader import *
from locate import *
from tracker import *
import open3d

meangrayr, meangrayc = 90.0/255, 88.0/255
widpatchr, widpatchr_r = 24, 17
widpatchc = 10
disr, disc = 70, 64

def do_init(data_pathL, data_pathR, system_path):
    locate_params = {'meangrayr': meangrayr,
                     'meangrayc': meangrayc,
                     'widpatchr': widpatchr,
                     'widpatchc': widpatchc,
                     'disr': disr,
                     'disc': disc}
    locate_params_r = {'meangrayr': meangrayr,
                     'meangrayc': meangrayc,
                     'widpatchr': widpatchr_r,
                     'widpatchc': widpatchc,
                     'disr': disr,
                     'disc': disc}

    ldata = DataLoader(data_pathL)
    base_name = 'hornDetect0618'
    spark_file_l = os.path.join('D:\\clean', '{}l.txt'.format(base_name))
    spark_file_r = os.path.join('D:\\clean', '{}r.txt'.format(base_name))
    ldata.load_horn_files(spark_file_l)
    rdata = DataLoader(data_pathR)
    rdata.load_horn_files(spark_file_r)
    template = LocateTemplate(locate_params['widpatchr'], locate_params['widpatchc'],
                              locate_params['meangrayr'], locate_params['meangrayc'],
                              locate_params['disr'], locate_params['disc'])
    template_r = LocateTemplate(locate_params_r['widpatchr'], locate_params_r['widpatchc'],
                              locate_params_r['meangrayr'], locate_params_r['meangrayc'],
                              locate_params_r['disr'], locate_params_r['disc'])
    area = LocateArea([0, 400, 600, 1900], [300, 550, 800, 1600])
    locate = Locate(LocateParams(template, area, [-1, -3.5], [1, -0.5]))
    locate_r = Locate(LocateParams(template_r, area, [3, -3], [1, 0.00001]))
    track = TrackerPAC(TrackParams(100, 50))
    track_r = TrackerPAC(TrackParams(100, 50))
    gt_file_l = os.path.join(system_path, 'gt')
    df_gt = ldata.load_gt_l(gt_file_l, 1000)# just for the first image
    gt_file_r = os.path.join(system_path, 'gt', 'R')
    df_gt_r = ldata.load_gt_l(gt_file_r, 1000)# just for the first image
    tgt_3d_horn = os.path.join(system_path, 'gt', 'mystd.ply')
    cloud = open3d.io.read_point_cloud(tgt_3d_horn)
    # open3d.visualization.draw_geometries([cloud], window_name="Open3D0")
    # cloud = PyntCloud.from_file(tgt_3d_horn)
    pan_points_3d = np.asarray(cloud.points)
    # x 列车行驶方向 [-16, 16] y 滑版方向[-700*, 700*] z垂直地面方向[23 ,-100]
    pan_points_3d[:, 1] = pan_points_3d[:, 1] - min(pan_points_3d[:, 1])
    pan_points_3d[:, 2] = pan_points_3d[:, 2] - min(pan_points_3d[:, 2])
    pan_length_tgt = max(pan_points_3d[:, 1]) - min(pan_points_3d[:, 1])
    pan_height_tgt = max(pan_points_3d[:, 2]) - min(pan_points_3d[:, 2])


    return ldata, locate, track, df_gt, rdata, locate_r, track_r, df_gt_r

def locate_contact(i, start_num, b_re_track, sum_time,
                   image_l, fImage_l, old_image,
                   df_gt, track, locate):
    img_idx = i + 1
    # loading gt position for the first images and init the patch_tgt for hash similarity evalution
    if i == start_num:
        temp_df = df_gt[df_gt['number'] == img_idx]
        points = temp_df[['point_l_x', 'point_l_y', 'point_2_x', 'point_2_y']].values.reshape(2, 2)
        points, ok, elapsed_t = track.do_track_refine(img_idx, image_l, image_l, points, b_re_init=True)
        patch_l_tgt, patch_r_tgt = get_patch(image_l, points[0:1, :]), get_patch(image_l, points[1:, :])
        track.params.update_patch(patch_l_tgt, patch_r_tgt)
    elif b_re_track:
        points, elapsed = locate.do_locate_refine(fImage_l, img_idx, 1)
        # update can be modifyed next by choosing a fixed hash value, for example < 20
        if verify_locate(locate.output_locate, img_idx):
            points, ok, elapsed_t = track.do_track_refine_hash(img_idx, image_l, image_l, points, b_re_init=True)
            if ok and max(track.hash) < 20:
                locate.do_update(fImage_l, img_idx)
            sum_time = sum_time + elapsed + elapsed_t
            b_re_track = False if ok else True
    else:
        points, ok, elapsed_t = track.do_track_refine_hash(img_idx, old_image, image_l)
        sum_time = sum_time + elapsed_t
        b_re_track = False if ok else True
        if not ok:
            points = track.get_simulate_points(img_idx)
    fps = 1. / (sum_time / (img_idx - start_num)) if sum_time != 0 else 0
    old_image = image_l
    return points, b_re_track, fps, old_image, sum_time, track, locate