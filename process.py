from locate import *
from tracker import *


def l_track(i, start_num, b_re_track, sum_time,
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
