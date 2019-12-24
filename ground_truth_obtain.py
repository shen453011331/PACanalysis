# load with_update.csv
# 以1000为一个基准，先进行批量读取，然后视频序列展示
# 若无大规模偏差，精细化点选，并保存为 001000.csv
# 若有大规模偏差，调节参数，重新生成该区域的结果。

# 实际调整操作为
# 1.批量展示图片，若无问题，回车下一帧
# 2.若有问题点esc，或者base数据集检测失败，跟随提示后，鼠标点击大图接触区域
# 3.esc后出现小图，点选两个接触点附近
# 4.esc后出现接触点和优化点，若红色定位准确，回车到下一帧
# 5.若红色不准确，esc，重新精确点选两个点，回车
import cv2
import pandas as pd
import numpy as np
from plotresult import *


class GT(object):
    def __init__(self, imgPath):
        self.path = imgPath

    def loadImage(self, img_num):
        image = cv2.imread('%s/%06d.bmp'%(self.path, img_num), cv2.IMREAD_GRAYSCALE)
        if img_num % 100 == 0:
            print('%d image has been processed' % img_num)
        return image

    def loadCsv(self, csv_path):
        self.df = pd.read_csv(csv_path + '4.groundtruth_base_multitrack.csv')

    def saveCsv(self, csv_path, num):
        self.tgt_file = '%s%06d.csv' % (csv_path, num)
        num_thousand = num/1000
        df = self.df.loc[self.df['number'] < num_thousand * 1000 + 1]
        df = df.loc[df['number'] > (num_thousand - 1) * 1000]
        df.to_csv(self.tgt_file, index=False)
        return df

    def changeCsv(self, image, df, df_all, img_num):
        # 第一步，将大小两个图画出来
        # 按回车，不改变，直接跳到下一张
        # 有问题，在大图上点选，重新生成小图
# 在小图上点选，自动匹配两个点，若ok，回车，保存修改
# 若不ok， esc， 重新手动在小图点选
        print('process No. %d image' % img_num)
        points = np.array([[df['point_l_x'].values[0], df['point_l_y'].values[0]],
                           [df['point_2_x'].values[0], df['point_2_y'].values[0]]], dtype='float')

        width = 200
        height = 200
        coordinate = np.floor(points.min(0) - np.array([width / 2, height / 2]))
        coordinates = [int(coordinate[0]), int(coordinate[1])]

        b_relocate = 0

        if coordinates[0] >= 0 and coordinates[1] >= 0:
            y1 = int(coordinates[1])
            y2 = int(coordinates[1] + height)
            x1 = int(coordinates[0])
            x2 = int(coordinates[0] + width)
            cv2.namedWindow("locate_large", 0)
            temp_image_large = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(temp_image_large, (x1, y1), (x2, y2), (255, 0, 0), 5)
            cv2.imshow("locate_large", temp_image_large)
            cv2.waitKey(1)
            image_slice = image[y1:y2, x1:x2]
            temp_points = points - coordinate
            cv2.namedWindow("locate", 0)
            temp_image = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
            cv2.circle(temp_image, (int(temp_points[0, 0]), int(temp_points[0, 1])), 5, (255, 255, 0), 1)
            cv2.circle(temp_image, (int(temp_points[1, 0]), int(temp_points[1, 1])), 5, (255, 255, 0), 1)
            cv2.imshow("locate", temp_image)
            key = cv2.waitKey(0)
            if key == 27:
                b_relocate = 1
                pass
            elif key == 13:
                pass
        else:
            b_relocate = 1
        if b_relocate:
            print('NO. %d locate error' % img_num)
            point_temp = []
            def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    loc = [x, y]
                    point_temp.append(x)
                    point_temp.append(y)

            loc = cv2.setMouseCallback('locate_large', on_EVENT_LBUTTONDOWN)
            cv2.imshow('locate_large', image)
            cv2.waitKey(0)
            coordinates = [int(point_temp[0] - width / 2), int(point_temp[1] - height / 2)]
            y1 = int(coordinates[1])
            y2 = int(coordinates[1] + height)
            x1 = int(coordinates[0])
            x2 = int(coordinates[0] + width)
            point_temp = []
            image_slice = image[y1:y2, x1:x2]
            loc = cv2.setMouseCallback('locate', on_EVENT_LBUTTONDOWN)
            cv2.imshow('locate', image_slice)
            cv2.waitKey(0)
            print(point_temp)
            fImage = cv2.normalize(image_slice.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            points = np.array([[point_temp[0], point_temp[1]],
                              [point_temp[2], point_temp[3]]], dtype='float')
            points_r = self.do_refine(fImage, points)
            print(points_r)
            temp_image = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
            cv2.circle(temp_image, (int(points[0, 0]), int(points[0, 1])), 2, (255, 255, 0), 1)
            cv2.circle(temp_image, (int(points[1, 0]), int(points[1, 1])), 2, (255, 255, 0), 1)
            cv2.circle(temp_image, (int(points_r[0, 0]), int(points_r[0, 1])), 2, (0, 0, 255), 1)
            cv2.circle(temp_image, (int(points_r[1, 0]), int(points_r[1, 1])), 2, (0, 0, 255), 1)
            cv2.imshow("locate", temp_image)
            key = cv2.waitKey(0)
            if key == 27:
                point_temp = []
                loc = cv2.setMouseCallback('locate', on_EVENT_LBUTTONDOWN)
                cv2.imshow("locate", image_slice)
                cv2.waitKey(0)
                points_r = np.array([[point_temp[0], point_temp[1]],
                                   [point_temp[2], point_temp[3]]], dtype='float')
            elif key == 13:
                pass
            num_csv = (img_num - 1) % 1000
            df_all.iloc[num_csv, 4] = points_r[0, 0] + x1
            df_all.iloc[num_csv, 2] = points_r[1, 0] + x1
            df_all.iloc[num_csv, 5] = points_r[0, 1] + y1
            df_all.iloc[num_csv, 3] = points_r[1, 1] + y1
        return df_all



    def do_refine(self, fImage, points):
        floatI = np.float32(fImage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(floatI, np.float32(points), (7, 7), (-1, -1), criteria)
        self.refinePoints = corners
        return self.refinePoints


if __name__ == '__main__':
    path = 'F:/dataset/L'
    csv_path = 'C:/Users/Administrator/Documents/sy_paper_contactAnalysis/new/'
    # 初始化类
    gt = GT(path)
    # 读取csv 和 图像文件
    gt.loadCsv(csv_path)
    # 视频序列展示
    num_thousand = 10
    num_big = num_thousand * 1000
    test_1 = 0  # test 1 using for check sequence is useful or not
    # if false do test 2, to verify each locating in frames
    if test_1:
        for img_num in range((num_thousand - 1) * 1000 + 1, num_thousand * 1000 + 1):
            # 读取图像
            image = gt.loadImage(img_num)
            # 获取指定df
            image_df = gt.df.loc[gt.df['number'] == img_num]
            # 提取左右点
            testShow = PlotResult()
            points = np.array([[image_df['point_l_x'].values[0], image_df['point_l_y'].values[0]],
                               [image_df['point_2_x'].values[0], image_df['point_2_y'].values[0]]], dtype='float')
            testShow.plot_locate_result(image, points, 1)

        # 若无大问题
        new_df = gt.saveCsv(csv_path, num_big)

    else:

        new_csv_file = '%s%06d.csv' % (csv_path, num_big)
        new_df = pd.read_csv(new_csv_file)
        # 点选修正
        startNum = 904
        # start num as least to be 1

        for img_num in range((num_thousand-1) * 1000 + startNum, num_thousand * 1000 + 1):
            # 读取图像
            image = gt.loadImage(img_num)
            # 获取指定df
            image_df = new_df.loc[new_df['number'] == img_num]

            new_df = gt.changeCsv(image, image_df, new_df, img_num)
            #将修正后的csv保存
            new_df.to_csv(new_csv_file, index=False)


