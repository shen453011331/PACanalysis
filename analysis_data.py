# 2020.1.2 this is file to analysis the input data
import numpy as np
import talib as ta
import pandas as pd
import math
import open3d as o3d

from process3D import *

class AnalysisData(object):
    def __init__(self, save_process, pro3d, plot):
        self.save = save_process
        self.pro3d = pro3d
        self.plot = plot
        self.df_gt_err = pd.DataFrame(columns=['number', 'l_err', 'r_err'])

    def compare_with_gt(self, df, point, index):
        point_gt = df[df['number'] == index]
        l_err = np.linalg.norm(np.array(point_gt[['point_l_x', 'point_l_y']].values) - point[0:1, :])
        r_err = np.linalg.norm(np.array(point_gt[['point_2_x', 'point_2_y']].values) - point[1:, :])
        temp_df = pd.DataFrame({'number': index, 'l_err': l_err, 'r_err': r_err}, index=[0])
        self.df_gt_err = pd.concat([self.df_gt_err, temp_df])

    def get_gt_error(self):
        err_1 = np.mean(self.df_gt_err['l_err'].values)
        err_2 = np.mean(self.df_gt_err['r_err'].values)
        return err_1, err_2

    def analysis_contact(self, df, index):
        df_now = df[df['number'] == index]
        l_p_l_re, l_p_r_re, l_p_l_lo, l_p_r_lo = self.get_left_point_from_df(df_now)
        dist1_l = self.get_dist1(l_p_l_re, l_p_r_re)
        dist2_l_l = self.get_dist2(l_p_l_lo, l_p_l_re)
        dist2_l_r = self.get_dist2(l_p_r_lo, l_p_r_re)
        df_seq = df[df['number'] <= index]
        dist3_l = 0
        speed_err_l = 0
        if len(df_seq) > 1:
            df_last = df[df['number'] == index - 1]
            l_p = np.mean(np.vstack([l_p_l_re, l_p_r_re]), axis=0)
            l_p_l_re_2, l_p_r_re_2, _, _ = self.get_left_point_from_df(df_last)
            l_p_2 = np.mean(np.vstack([l_p_l_re_2, l_p_r_re_2]), axis=0)
            dist3_l = self.get_dist3(l_p, l_p_2)
        if len(df_seq) > 5:
            l_p_l_re_seq, l_p_r_re_seq, _, _ = self.get_left_point_from_df(df_seq)
            l_p_x = np.mean(np.hstack([l_p_l_re_seq[:, 0:1], l_p_r_re_seq[:, 0:1]]), axis=1)
            l_p_y = np.mean(np.hstack([l_p_l_re_seq[:, 1:], l_p_r_re_seq[:, 1:]]), axis=1)
            l_p_seq = np.vstack([l_p_x, l_p_y]).T
            speed_err_l = self.get_speed_error(l_p_seq)
        return dist1_l, dist2_l_l, dist2_l_r, dist3_l, speed_err_l

    def get_dist1(self, pl, pr):
        # this to get distance for two contact points after refine in one frame
        dist = np.abs(pl[0, 0] - pr[0, 0])
        return dist

    def get_dist2(self, p_lo, p_re):
        # this is to get distance for contact points before and after refine
        dist = np.linalg.norm(p_lo-p_re)
        return dist

    def get_dist3(self, p_i, p_iplus1):
        # this is to get distance for contact points before and after refine
        dist = np.linalg.norm(p_i-p_iplus1)
        return dist

    def get_speed_error(self, p_seq):
        speed = p_seq[1:, :] - p_seq[:-1, :]
        speed = speed[-4:, :]
        speed_norm = np.linalg.norm(speed, axis=1)
        speed_norm = speed_norm.astype(np.float64)
        # float_data = [float(x) for x in speed_norm]
        speed_ma = ta.MA(speed_norm, 3)
        return np.abs(speed_norm[-1]-speed_ma[-1])

    def get_left_point_from_df(self, df):
        l_p_l_re = df[['l_p_l_re_x', 'l_p_l_re_y']].values
        l_p_r_re = df[['l_p_r_re_x', 'l_p_r_re_y']].values
        l_p_l_lo = df[['l_p_l_lo_x', 'l_p_l_lo_y']].values
        l_p_r_lo = df[['l_p_r_lo_x', 'l_p_r_lo_y']].values
        return l_p_l_re, l_p_r_re, l_p_l_lo, l_p_r_lo


# this function is used for get the values for calculate for smoothness value
# there are 3dpoints [contact points, lhorn, rhorn]
# image locateion in [left right]  [left contact right contact lhorn rhorn]
# deep processing  pantagraph head pose[theta, lachu value, height]
# moving values [3d contact point] [mean 1, 3, 5]
# spark detect
def get_smooth_values(ldata, rdata, output_3d,
                      locate_output, locate_output_r, track_output_r, track_output):
    df_column = ['idx', 'path_l', 'path_r',
                 'p3d_lhorn_x', 'p3d_lhorn_y', 'p3d_lhorn_z',
                 'p3d_rhorn_x', 'p3d_rhorn_y', 'p3d_rhorn_z',
                 'points_3d_x', 'points_3d_y', 'points_3d_z',
                 'l_points_l_x', 'l_points_l_y', 'l_points_r_x', 'l_points_r_y',
                 'l_lhorn_x', 'l_lhorn_y', 'l_rhorn_x', 'l_rhorn_y',  # points location in left images
                 'r_points_l_x', 'r_points_l_y', 'r_points_r_x', 'r_points_r_y',
                 'r_lhorn_x', 'r_lhorn_y', 'r_rhorn_x', 'r_rhorn_y',  # points location in right images
                 'theta', 'distance', 'height',
                 # this height is the value of the height of the center of the pantograph head
                 'move_1', 'move_3', 'move_5',
                 'spark', 'spark_x', 'spark_y' # spark 0/1 spark_x,y if spark is 0 set as 0
                 ]
    df = pd.DataFrame(columns=df_column)
    result_3d = output_3d.result_list

    for i in range(0, 21732):
        img_idx = i + 1
        path_l = '%s/%06d.bmp' % (ldata.data_path, img_idx)
        path_r = '%s/%06d.jpg' % (rdata.data_path, img_idx)

        p3d_lhorn, p3d_rhorn, points_3d = result_3d[i]['p3d_lhorn'][:,0], \
                                          result_3d[i]['p3d_rhorn'][:,0], result_3d[i]['points_3d'][:,0]
        distance, theta = result_3d[i]['distance'], result_3d[i]['theta']
        height = (p3d_lhorn[2] + p3d_rhorn[2])/2 + result_3d[i]['height']

        points, points_r = get_image_points_locations(locate_output, locate_output_r,
                                                      track_output_r, track_output, img_idx)

        lhorn_l, lhorn_r = ldata.load_hornPoints_from_df(ldata.df, i + 1)
        rhorn_l, rhorn_r = rdata.load_hornPoints_from_df(rdata.df, i + 1)

        move_1, move_3, move_5 = get_move(output_3d, i)
        spark, spark_x, spark_y = get_spark_points(ldata.df, rdata.df, img_idx)
        df_dict = {'idx': img_idx, 'path_l': path_l, 'path_r': path_r,
                   'p3d_lhorn_x':p3d_lhorn[0], 'p3d_lhorn_y':p3d_lhorn[1], 'p3d_lhorn_z':p3d_lhorn[2],
                   'p3d_rhorn_x':p3d_rhorn[0], 'p3d_rhorn_y':p3d_rhorn[1], 'p3d_rhorn_z':p3d_rhorn[2],
                   'points_3d_x':points_3d[0], 'points_3d_y':points_3d[1], 'points_3d_z':points_3d[2],
                   'l_points_l_x':points[0, 0], 'l_points_l_y':points[0, 1],
                   'l_points_r_x':points[1, 0], 'l_points_r_y':points[1, 1],
                   'l_lhorn_x':lhorn_l[0], 'l_lhorn_y':lhorn_l[1], 'l_rhorn_x':lhorn_r[0], 'l_rhorn_y':lhorn_r[1],
                   'r_points_l_x':points_r[0, 0], 'r_points_l_y':points_r[0, 1],
                   'r_points_r_x':points_r[1, 0], 'r_points_r_y':points_r[1, 1],
                   'r_lhorn_x':rhorn_l[0], 'r_lhorn_y':rhorn_l[1], 'r_rhorn_x':rhorn_r[0], 'r_rhorn_y':rhorn_r[1],
                   'theta': theta, 'distance': distance, 'height': height,
                   'move_1': move_1, 'move_3':move_3, 'move_5': move_5,
                   'spark':spark, 'spark_x':spark_x, 'spark_y':spark_y}
        if img_idx%1000 == 0:
            print('save {} imgae result'.format(img_idx))
        df = df.append([df_dict], ignore_index=True)

    return df


# using locate and track result data to get locate points
def get_image_points_locations(locate_output, locate_output_r, track_output_r, track_output, img_idx):
    dict_data = None
    points_zero = np.zeros((2, 2), dtype=np.float32)
    if img_idx in locate_output.index_list:
        index = locate_output.index_list.index(img_idx)
        dict_data = locate_output.result_list[index]
    elif img_idx in track_output.index_list:
        index = track_output.index_list.index(img_idx)
        dict_data = track_output.result_list[index]
    points = np.vstack((dict_data['points_l'], dict_data['points_r'])) \
        if dict_data is not None else points_zero

    points = verify_points(points)
    dict_data_r = None
    if img_idx in locate_output_r.index_list:
        index = locate_output_r.index_list.index(img_idx)
        dict_data_r = locate_output_r.result_list[index]
    elif img_idx in track_output_r.index_list:
        index = track_output_r.index_list.index(img_idx)
        dict_data_r = track_output_r.result_list[index]
    points_r = np.vstack((dict_data_r['points_l'], dict_data_r['points_r'])) \
        if dict_data_r is not None else points_zero
    points_r = verify_points(points_r)
    return points, points_r


def verify_points(points):
    # filter out the poins which is not in [900,2336]
    if np.max(points[:,0]) > 2336 or np.min(points[:,0]) <0 or np.max(points[:,1]) > 900 or np.min(points[:,1]) <0:
        points = np.zeros((2, 2), dtype=np.float32)
    return points



def get_move(output_3d, i):
    start_num = 0
    move_1 = output_3d.result_list[i]['distance'] - output_3d.result_list[i-1]['distance']\
        if i - start_num > 0 else 0
    move_3 = output_3d.result_list[i]['distance'] - output_3d.result_list[i-3]['distance']\
        if i - start_num > 2 else 0
    move_3 = move_3/3
    move_5 = output_3d.result_list[i]['distance'] - output_3d.result_list[i-5]['distance']\
        if i - start_num > 4 else 0
    move_5 = move_5/5
    return move_1, move_3, move_5


def get_spark_points(df_l, df_r, img_idx):
    path = '/VOC2007L/JPEGImages/%06d.jpg' % img_idx
    temp_df = df_l[df_l['image_path'].isin([path])]
    temp2 = temp_df[temp_df['left/right'] == 3]
    if len(temp2) !=0:
        return 1, temp2.x1.values, temp2.y2.values
    else:
        temp_df = df_r[df_r['image_path'].isin([path])]
        temp2 = temp_df[temp_df['left/right'] == 3]
        if len(temp2) != 0:
            return 1, temp2.x1.values, temp2.y2.values
        else:
            return 0, 0, 0


def get_horn_points_seq(start_num, output_3d, pan_points_3d, pan_length_tgt, pan_height_tgt):
    # this function is used to get a point cloud which the y sequence is the img_idx to show the pan head move with time
    for i in range(start_num, 10000, 10):  # args.data_num):
        img_idx = i + 1
        if img_idx % 1000 == 0:
            print('image No. {}'.format(img_idx))
        # plot 3D pantograph
        theta = output_3d.result_list[i]['theta']
        length = output_3d.result_list[i]['width']
        height = output_3d.result_list[i]['height']
        p3d_lhorn = output_3d.result_list[i]['p3d_lhorn']
        if 0 not in p3d_lhorn:
            # temp_dic = {'idx': img_idx, 'p3d_lhorn_x': p3d_lhorn[0],
            #             'p3d_lhorn_y': p3d_lhorn[1], 'p3d_lhorn_z': p3d_lhorn[2]}
            # temp_df = pd.DataFrame(temp_dic, index={0}, columns=['idx', 'p3d_lhorn_x', 'p3d_lhorn_y', 'p3d_lhorn_z'])
            # df = df.append(temp_df)
            rotate_matrix = np.array([[math.cos(theta), math.sin(theta)],
                                      [-math.sin(theta), math.cos(theta)]])
            temp_pan_3d = pan_points_3d.copy()
            # time sequence
            temp_pan_3d[:, 0] = temp_pan_3d[:, 0] + i / 10
            # zoom
            temp_pan_3d[:, 1] = temp_pan_3d[:, 1] * length / pan_length_tgt
            temp_pan_3d[:, 2] = temp_pan_3d[:, 2] * height / pan_height_tgt
            # rotate
            temp_pan_3d[:, 1:] = np.dot(rotate_matrix, pan_points_3d[:, 1:].T).T
            # shift
            temp_pan_3d[:, 0] = temp_pan_3d[:, 0] + np.squeeze(p3d_lhorn)[1]
            temp_pan_3d[:, 1] = temp_pan_3d[:, 1] + np.squeeze(p3d_lhorn)[0]
            temp_pan_3d[:, 2] = temp_pan_3d[:, 2] + np.squeeze(p3d_lhorn)[2]

            # test adding color to points cloud
            points = np.array([[0.1, 0.1, 0.1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            colors = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

            test_pcd = o3d.geometry.PointCloud()  # 定义点云
            test_pcd.points = o3d.utility.Vector3dVector(points)  # 定义点云坐标位置
            test_pcd.colors = o3d.Vector3dVector(colors)

            pan_point_seq = temp_pan_3d if pan_point_seq is None else np.vstack([pan_point_seq, temp_pan_3d])
    return pan_point_seq


def visulize_point_cloud(pan_point_seq):
    # this function is used to visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pan_point_seq)
    o3d.visualization.draw_geometries([point_cloud], window_name="Open3D0")


def get_smoothness_value(df, idx, weight=[0.6, 0.4, 0, 0]):
    # smoothness value is [0,1] 0 is good 1 is bad

    df_temp = df[idx:idx+1]
    # #1 spark detect
    if df_temp['spark'].values[0] == 1:
        return 1, 0, [0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    else:
        # contact lachu value
        p_lachu = get_gaussion_value_from_data(df_temp['distance'].values[0], [-400, 600], 0.2)
        # pose theta height
        p_theta = get_gaussion_value_from_data(df_temp['theta'].values[0], [2.8, 3.2])
        p_h = get_gaussion_value_from_data(df_temp['height'].values[0], [3490, 4500],0.05)
        # image location
        # left image
        p_c_1 = get_gaussion_value_from_data(df_temp['l_points_l_x'].values[0], [900, 1500])
        p_c_2 = get_gaussion_value_from_data(df_temp['l_points_r_x'].values[0], [900, 1500])

        p_c_1_y = get_gaussion_value_from_data(df_temp['r_points_l_y'].values[0], [325, 460])
        p_c_2_y = get_gaussion_value_from_data(df_temp['r_points_r_y'].values[0], [325, 460])
        # right image
        p_c_3 = get_gaussion_value_from_data(df_temp['r_points_l_x'].values[0], [900, 1600])
        p_c_4 = get_gaussion_value_from_data(df_temp['r_points_r_x'].values[0], [900, 1600])

        p_c_3_y = get_gaussion_value_from_data(df_temp['r_points_l_y'].values[0], [340, 475])
        p_c_4_y = get_gaussion_value_from_data(df_temp['r_points_r_y'].values[0], [340, 475])
        w_lachu, w_pose, w_image_x, w_image_y = weight
        p_smooth = (w_lachu * p_lachu + w_pose * np.mean([p_theta, p_h])
                    + w_image_x * np.mean([p_c_1, p_c_2, p_c_3, p_c_4])
                    + w_image_y * np.mean([p_c_1_y, p_c_2_y, p_c_3_y, p_c_4_y])) / np.sum([w_lachu, w_pose,
                                                                                           w_image_x, w_image_y])
        return p_smooth, p_lachu, [p_theta, p_h], [p_c_1, p_c_2, p_c_3, p_c_4], [p_c_1_y, p_c_2_y, p_c_3_y, p_c_4_y]


# def show_smooth_fault(df,i, p_conf):
#     p_smooth, p_lachu, [p_theta, p_h], [p_c_1, p_c_2, p_c_3, p_c_4], [p_c_1_y, p_c_2_y, p_c_3_y,
#                                                                       p_c_4_y] = get_smoothness_value(df, i)
#     if p_smooth ==1 or p_conf == 1:
#             if 1> p_smooth >0.6:
#                 img_idx = i + 1
#                 print('------------------------------------------------')
#                 print('idx:{},lachu:{}, theta:{},h:{},c_x:{},{},{},{}'.format(i + 1, p_lachu, p_theta, p_h,
#                                                                               p_c_1, p_c_2, p_c_3, p_c_4))
#                 print('c_y:{},{},{},{}'.format(p_c_1_y, p_c_2_y, p_c_3_y, p_c_4_y))
#                 print('polar value is {}'.format(k))
#                 image_l, fImage_l, _, _ = ldata.load(img_idx)
#                 image_r, fImage_r, _, _ = rdata.load_R(img_idx)
#                 points_l = np.vstack([df[i:i + 1][['l_points_l_x', 'l_points_l_y']].values[0],
#                                       df[i:i + 1][['l_points_r_x', 'l_points_r_y']].values[0]])
#                 points_r = np.vstack([df[i:i + 1][['r_points_l_x', 'r_points_l_y']].values[0],
#                                       df[i:i + 1][['r_points_r_x', 'r_points_r_y']].values[0]])
#
#                 print('points l')
#                 print(points_l)
#                 print('points r')
#                 print(points_r)
#                 image_l = show.circle_points_on_image(image_l, points_l)
#                 image_r = show.circle_points_on_image(image_r, points_l)
#                 image_show = np.hstack([image_l, image_r])
#                 cv2.putText(image_show,'#{}'.format(img_idx), (100,100), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 12)
#                 cv2.imshow(window_name, image_show)
#                 cv2.waitKey()
#     cv2.destroyAllWindows()


def get_gaussion_value_from_data(value, thre, range_thre=0.1):
    # p1 p2 is a area which inside the output value is 0 named as safe area
    # sigma and mean is for calculate guassian but our output is 1 - gaussian
    # if input value sets is in [e1,e2] which [p1,p2] set as [e1 + (e2-e1)*0.1,e2 - (e2-e1)*0.1]
    # sigma is (e2-e1)*0.1/6
    e1, e2 = thre
    p1, p2 = e1 + (e2-e1)*range_thre, e2 - (e2-e1)*range_thre
    sigma = (e2-e1)*range_thre/6
    if p1 < value < p2:
        return 0
    elif np.isnan(value):
        return 1
    else:
        x = np.min([np.abs(value-p1), np.abs(value-p2)])
        return 1 - normal_distribution(x, 0, sigma)/normal_distribution(0, 0, sigma)


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def get_locate_confidence(df, idx):
    # points is zero or not
    i = idx
    points_l = np.vstack([df[i:i + 1][['l_points_l_x', 'l_points_l_y']].values[0],
                          df[i:i + 1][['l_points_r_x', 'l_points_r_y']].values[0]])
    points_r = np.vstack([df[i:i + 1][['r_points_l_x', 'r_points_l_y']].values[0],
                          df[i:i + 1][['r_points_r_x', 'r_points_r_y']].values[0]])
    if points_l.min() <=0 or points_r.min() <=0:
        return 0, 0
    if points_l[:, 0].min() < 900 or points_l[:, 0].max() > 1500:
        return 0, 0
    if points_r[:, 0].min() < 900 or points_r[:, 0].max() > 1600:
        return 0, 0
    if points_l[:, 1].min() < 325 or points_l[:, 1].max() > 460:
        return 0, 0
    if points_r[:, 1].min() < 340 or points_r[:, 1].max() > 475:
        return 0, 0
    pro3d = Process3D()
    lines, k = pro3d.getPolarLine(np.mean(points_l, 0), np.mean(points_r, 0))
    if not -3 < k < 3:
        return 0, k
    if abs(df[i:i + 1]['move_1'].values[0]) > 40:
        return 0, k
    return 1, k