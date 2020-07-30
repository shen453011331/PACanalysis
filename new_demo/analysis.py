import pandas as pd
import numpy as np
import math


class SmoothAnalysis(object):
    def __init__(self):
        self.df_column = ['idx', 'path_l', 'path_r',
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
        self.df = pd.DataFrame(columns=self.df_column)

        self.distance_list = []

    def get_smooth_para_df(self, img_idx, ldata, rdata,
                           p3d_lhorn, p3d_rhorn, points_3d,
                           points, points_r,
                           lhorn_l, lhorn_r, rhorn_l, rhorn_r,
                           distance, theta, height):
        path_l = '%s/%06d.bmp' % (ldata.data_path, img_idx)
        path_r = '%s/%06d.jpg' % (rdata.data_path, img_idx)
        height = (p3d_lhorn[2] + p3d_rhorn[2])/2 + height
        self.distance_list.append(distance)
        move_1, move_3, move_5 = get_move(img_idx-1, self.distance_list)
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
        self.df = self.df.append([df_dict], ignore_index=True)

    def get_smoothness_value(self, idx, weight=[0.6, 0.4, 0, 0]):
            # smoothness value is [0,1] 0 is good 1 is bad
        df_temp = self.df[self.df['idx'] == idx]
        # #1 spark detect
        if df_temp['spark'].values[0] == 1:
            return 1, 0, [0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        else:
            # contact lachu value
            p_lachu = get_gaussion_value_from_data(df_temp['distance'].values[0], [-400, 600], 0.2)
            # pose theta height
            p_theta = get_gaussion_value_from_data(df_temp['theta'].values[0], [2.8, 3.2])
            p_h = get_gaussion_value_from_data(df_temp['height'].values[0], [3490, 4500], 0.05)
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
                                                                                               w_image_x,
                                                                                               w_image_y])
            return p_smooth, p_lachu, [p_theta, p_h], [p_c_1, p_c_2, p_c_3, p_c_4], [p_c_1_y, p_c_2_y, p_c_3_y,
                                                                                         p_c_4_y]

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


def get_move(i, distance_list):
    start_num = 0
    move_1 = distance_list[-1] - distance_list[-2] if i - start_num > 0 else 0
    move_3 = distance_list[-1] - distance_list[-4] if i - start_num > 2 else 0
    move_3 = move_3/3
    move_5 = distance_list[-1] - distance_list[-6] if i - start_num > 4 else 0
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