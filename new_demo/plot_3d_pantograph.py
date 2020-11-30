import argparse
import cv2
import time
import os
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
from basic_function import *

from analysis_data import *
from dataloader import *



parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_pathL',  dest='data_pathL', type=str, default='F:/dataset/L', help='path of data')
parser.add_argument('--data_pathR',  dest='data_pathR', type=str, default='F:/dataset/R/JPEGImages', help='path of data')
parser.add_argument('--title',  dest='title', type=str, default='full_position_lr', help='path of data')
parser.add_argument('--system_path',  dest='system_path', type=str, default='D:/clean/PACanalysis/', help='path of data')
args = parser.parse_args()


if __name__ == '__main__':
    # load process results
    save_path = os.path.join(args.system_path, 'result', args.title)
    with open(os.path.join(save_path, 'result_3d.pkl'), 'rb') as file:
        output_3d = pickle.loads(file.read())
    # load pantograph points in one line
    tgt_3d_horn = os.path.join(args.system_path, 'gt', 'mystd.ply')
    output_file = os.path.join(args.system_path, 'output', args.title, 'horn3d_colors.ply')
    cloud = o3d.io.read_point_cloud(tgt_3d_horn)
    pan_points_3d = np.asarray(cloud.points)
    # x 列车行驶方向 [-16, 16] y 滑版方向[-700*, 700*] z垂直地面方向[23 ,-100]
    pan_points_3d[:, 1] = pan_points_3d[:, 1] - min(pan_points_3d[:, 1])
    pan_points_3d[:, 2] = pan_points_3d[:, 2] - min(pan_points_3d[:, 2])
    pan_length_tgt = max(pan_points_3d[:, 1]) - min(pan_points_3d[:, 1])
    pan_height_tgt = max(pan_points_3d[:, 2]) - min(pan_points_3d[:, 2])
    pan_points_3d = pan_points_3d[(pan_points_3d[:, 0] < 1 / 2) & (pan_points_3d[:, 0] > -1 / 2), :]
    # pan_point_seq = None
    # 普通用弓的颜色   通过load 不同txt 实现不同过渡色
    color_file = os.path.join(args.system_path, 'new_demo', 'winter.txt')
    color_sample = np.loadtxt(color_file,delimiter = ',')
    colors = None
    for i in range(pan_points_3d.shape[0]):
        tmp_scalar = ( pan_points_3d[i, 2] - min(pan_points_3d[:, 2])) / pan_height_tgt
        tmp_color = get_color(tmp_scalar, color_sample)
        # tmp_color = color_sample[int(tmp_scalar*64)] * 255
        # tmp_color = tmp_color.astype(int)
        colors = np.vstack([colors, tmp_color]) if colors is not None else tmp_color
    # 异常用弓颜色
    # colors = np.expand_dims([0, 191, 255], 0).repeat(pan_points_3d.shape[0], axis=0)
    color_alarm = np.expand_dims([190, 0, 0], 0).repeat(pan_points_3d.shape[0], axis=0)
    # # show
    point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(pan_points_3d)
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([point_cloud], window_name="Open3D0")

    theta = []
    height = []
    for i in range(16000):
        p3d_lhorn = output_3d.result_list[i]['p3d_lhorn']
        if 0 not in p3d_lhorn:
            theta.append(output_3d.result_list[i]['theta'])
            height.append(np.squeeze(p3d_lhorn)[2])
    ##求一下均值和方差，用来看相应的颜色偏移量是多少比较合适
    # 2020/10/17 15：41
    theta = np.array(theta)
    mean_theta = np.mean(theta)
    print(mean_theta)
    std_theta = np.std(theta)
    print(std_theta)
    scalar_theta = np.abs(theta - mean_theta)/std_theta
    max_scalar_theta = np.max(scalar_theta)
    height = np.array(height)
    mean_h = np.mean(height)
    print(mean_h)
    std_h = np.std(height)
    print(std_h)
    scalar_h = np.abs(height - mean_h)/std_h
    max_scalar_h = np.max(scalar_h)
    # plt.plot(height)
    # plt.show()
    # tmp_color_scalr = [0.5 if np.abs(h - mean_h) / std_h < 1 else np.abs(
    #     h - mean_h) / std_h / max_scalar_h * 0.5 + 0.5 for h in height]
    # plt.plot(tmp_color_scalr)
    # plt.show()
    # tmp_color_scalr = [0.5 if np.abs(i - mean_theta) / std_theta < 1 else np.abs(
    #     i - mean_theta) / std_theta / max_scalar_theta * 0.5 + 0.5 for i in theta]
    #
    # # theta = [output_3d.result_list[i]['theta'] for i in range(16000)]
    # plt.plot(tmp_color_scalr)
    # plt.show()
    pan_point_seq = None
    color_seq = None
    for i in range(16000):
        theta = output_3d.result_list[i]['theta']
        length = output_3d.result_list[i]['width']
        height = output_3d.result_list[i]['height']
        p3d_lhorn = output_3d.result_list[i]['p3d_lhorn']
        tmp_3d_points = get_ith_points(pan_points_3d, p3d_lhorn, theta, height, length,
                                       pan_length_tgt, pan_height_tgt, i)
        if tmp_3d_points is not None:
            # 根据theta 对异常值进行判断
            tmp_color_scalr_theta = 0.5 if np.abs(theta - mean_theta)/std_theta < 1 else np.abs(theta - mean_theta)/std_theta/max_scalar_theta * 0.5 + 0.5
            h = np.squeeze(p3d_lhorn)[2]
            tmp_color_scalr_h = 0.5 if np.abs(h - mean_h) / std_h < 1 else np.abs(h - mean_h) / std_h / max_scalar_h * 0.5 + 0.5
            tmp_color_scalr = np.max([tmp_color_scalr_theta,tmp_color_scalr_h])
            tmp_color = colors if tmp_color_scalr < 0.7 else color_alarm
            # tmp_color_scalr = 0.5 if np.abs(theta - mean_theta)/std_theta < 1 else np.abs(theta - mean_theta)/std_theta/max_scalar_theta * 0.5 + 0.5
            # tmp_color = colors * tmp_color_scalr
            pan_point_seq = tmp_3d_points if pan_point_seq is None else np.vstack([pan_point_seq, tmp_3d_points])
            color_seq = tmp_color if color_seq is None else np.vstack([color_seq, tmp_color])
            if i % 1000 == 1:
                print('No.{}'.format(i))
            #     point_cloud.points = o3d.utility.Vector3dVector(pan_point_seq)
            #     point_cloud.colors = o3d.utility.Vector3dVector(color_seq)
            #     o3d.visualization.draw_geometries([point_cloud], window_name="Open3D0")
    point_cloud.points = o3d.utility.Vector3dVector(pan_point_seq)
    point_cloud.colors = o3d.utility.Vector3dVector(color_seq)
    o3d.visualization.draw_geometries([point_cloud], window_name="Open3D0")
    o3d.io.write_point_cloud(output_file, point_cloud)

    # show temp sequence slice
    # temp_points = os.path.join(args.system_path, 'output', args.title, 'horn3d_colors.ply')
    # cloud = o3d.io.read_point_cloud(temp_points)
    # pan_points_3d = np.asarray(cloud.points)
    # colors_3d = np.asarray(cloud.colors)
    # start, end = 10000, 12000
    # pan_tmp_3d = pan_points_3d[(pan_points_3d[:, 0] < 1 / 2 + end) & (pan_points_3d[:, 0] > -1 / 2 + start), :]
    # colors_3d = colors_3d[(pan_points_3d[:, 0] < 1 / 2 + end) & (pan_points_3d[:, 0] > -1 / 2 + start), :]
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(pan_tmp_3d)
    # point_cloud.colors = o3d.utility.Vector3dVector(colors_3d)
    # o3d.visualization.draw_geometries([point_cloud], window_name="Open3D0")
    # output_file = os.path.join(args.system_path, 'output', args.title, 'horn3d_colors-10000-12000.ply')
    # o3d.io.write_point_cloud(output_file, point_cloud)

    # the two image are in full_position_lr with html
    # also can change the value in scatter_h and scatter_theta1.txt

    # get_images(np.linspace(10001, 12000, 2000), theta[10001:12000],'theta_10000-12000.html','theta')
    # data = np.vstack([np.linspace(10001, 12000, 2000), theta[10001:12001]])
    # np.savetxt('scatter_theta1.txt', data.T, delimiter= ',', fmt= '%.4f')
    # data = np.vstack([np.linspace(10001, 12000, 2000), height[10001:12001]])
    # np.savetxt('scatter_h.txt', data.T, delimiter= ',', fmt= '%.4f')

