import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import Line, Page
from pyecharts.commons.utils import JsCode
from pyecharts.faker import Collector, Faker


class PlotResult(object):
    def __init__(self):
        self.boolLocateshow = 1
        pass

    def plot_locate_result(self, image, points, wait_time=1):
        if self.boolLocateshow:
            width = 200
            height = 200
            coordinate = np.floor(points.min(0)-np.array([width/2, height/2]))
            coordinates = [int(coordinate[0]), int(coordinate[1])]
            if coordinates[0] >= 0 and coordinates[1] >= 0:
                y1 = int(coordinates[1])
                y2 = int(coordinates[1] + height)
                x1 = int(coordinates[0])
                x2 = int(coordinates[0] + width)
                cv2.namedWindow("locate_large",0)
                temp_image_large = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(temp_image_large, (x1, y1), (x2, y2), (255, 0, 0), 5)
                cv2.imshow("locate_large", temp_image_large)
                cv2.waitKey(wait_time)
                image_slice = image[y1:y2, x1:x2]
                temp_points = points - coordinate
                cv2.namedWindow("locate", 0)
                temp_image = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
                cv2.circle(temp_image, (int(temp_points[0, 0]), int(temp_points[0, 1])), 5, (255, 255, 0), 1)
                cv2.circle(temp_image, (int(temp_points[1, 0]), int(temp_points[1, 1])), 5, (255, 255, 0), 1)
                cv2.imshow("locate", temp_image)
                cv2.waitKey(wait_time)

    def plot_locate_refine_result(self, image, points, points_l, waittime=1):
        if self.boolLocateshow:
            width = 100
            height = 100
            coordinate = np.floor(points.min(0) - np.array([width / 2, height / 2]))
            coordinates = [int(coordinate[0]), int(coordinate[1])]
            if coordinates[0] >= 0 and coordinates[1] >= 0:
                y1 = int(coordinates[1])
                y2 = int(coordinates[1] + height)
                x1 = int(coordinates[0])
                x2 = int(coordinates[0] + width)
                cv2.namedWindow("locate_large", 0)
                temp_image_large = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(temp_image_large, (x1, y1), (x2, y2), (255, 0, 0), 5)
                cv2.imshow("locate_large", temp_image_large)
                cv2.waitKey(waittime)
                image_slice = image[y1:y2, x1:x2]
                temp_points = points - coordinate
                cv2.namedWindow("locate", 0)
                temp_image = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
                cv2.circle(temp_image, (int(temp_points[0, 0]), int(temp_points[0, 1])), 2, (255, 255, 0), 1)
                cv2.circle(temp_image, (int(temp_points[1, 0]), int(temp_points[1, 1])), 2, (255, 255, 0), 1)
                temp_points_l = points_l - coordinate
                cv2.circle(temp_image, (int(temp_points_l[0, 0]), int(temp_points_l[0, 1])), 2, (0, 0, 255), 1)
                cv2.circle(temp_image, (int(temp_points_l[1, 0]), int(temp_points_l[1, 1])), 2, (0, 0, 255), 1)
                cv2.imshow("locate", temp_image)
                cv2.waitKey(waittime)
                a = 0

    def plot_test_patch(self, new_patch, bbox_local_l, bbox_local_r):
        cv2.namedWindow('testPatch', 0)
        temp_new_patch = cv2.cvtColor(new_patch, cv2.COLOR_GRAY2BGR)
        y1 = int(bbox_local_l[1])
        y2 = int(bbox_local_l[1] + bbox_local_l[3])
        x1 = int(bbox_local_l[0])
        x2 = int(bbox_local_l[0] + bbox_local_l[2])
        cv2.rectangle(temp_new_patch, (x1, y1), (x2, y2), (255, 0, 0), 5)
        y1 = int(bbox_local_r[1])
        y2 = int(bbox_local_r[1] + bbox_local_r[3])
        x1 = int(bbox_local_r[0])
        x2 = int(bbox_local_r[0] + bbox_local_r[2])
        cv2.rectangle(temp_new_patch, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.imshow('testPatch', temp_new_patch)
        cv2.waitKey(0)

    def plot_two_images(self, limage, rimage):
        cv2.namedWindow('left and right', 0)
        mergeImage = np.hstack([limage, rimage])
        cv2.imshow('left and right', mergeImage)
        cv2.waitKey(0)

    def plot_points_on_two_images(self, image_l, image_r, l_points, r_points):
        temp_image_l = self.circle_points_on_image(image_l, l_points)
        temp_image_r = self.circle_points_on_image(image_r, r_points)
        self.plot_two_images(temp_image_l, temp_image_r)

    def circle_points_on_image(self, image, points):
        # points are N*2 [x,y] numpy
        temp_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(points.shape[0]):
            cv2.circle(temp_image, (int(points[i, 0]), int(points[i, 1])), 5, (255, 255, 0), 2)
        return temp_image

    def load_plot_data_full(self, csv_files):
        # get distance between two points
        df = pd.read_csv(csv_files)
        x_data = df['number'].values
        x1 = df['point_l_x'].values
        y1 = df['point_l_y'].values
        x2 = df['point_2_x'].values
        y2 = df['point_2_y'].values
        return x_data, x1, y1, x2, y2


    def load_plot_data(self, csv_files):
        # get distance between two points
        df = pd.read_csv(csv_files)
        x_data = df['number'].values.tolist()
        y1 = df['point_l_x'].values
        y2 = df['point_2_x'].values
        y_data = np.abs(y2-y1).tolist()
        return x_data, y_data

    def load_plot_data_2(self, csv_files):
        # get distance between points in sequence
        df = pd.read_csv(csv_files)
        x_data = df['number'].values[1:].tolist()
        y1 = df['point_l_x'].values
        y2 = df['point_2_x'].values

        y_data_1 = np.around(np.abs(y1[:-1] - y1[1:]), decimals=2).tolist()
        y_data_2 = np.around(np.abs(y2[:-1] - y2[1:]), decimals=2).tolist()
        return x_data, y_data_1, y_data_2

    def plot_lines(self, x_data, y_data, filename):
        background_color_js = (
            "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
            "[{offset: 0, color: '#c86589'}, {offset: 1, color: '#06a7ff'}], false)"
        )
        area_color_js = (
            "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
            "[{offset: 0, color: '#eb64fb'}, {offset: 1, color: '#3fbbff0d'}], false)"
        )
        c = (
            Line(init_opts=opts.InitOpts(bg_color=JsCode(background_color_js)))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
                series_name="l_r_corner_dist",
                y_axis=y_data,
                is_smooth=True,
                is_symbol_show=True,
                symbol="circle",
                symbol_size=6,
                linestyle_opts=opts.LineStyleOpts(color="#fff"),
                label_opts=opts.LabelOpts(is_show=True, position="top", color="white"),
                itemstyle_opts=opts.ItemStyleOpts(
                    color="red", border_color="#fff", border_width=3
                ),
                tooltip_opts=opts.TooltipOpts(is_show=False),
                areastyle_opts=opts.AreaStyleOpts(color=JsCode(area_color_js), opacity=1),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="show distance between two points in one image",
                    pos_top="5%",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(color="#fff", font_size=16),
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    boundary_gap=False,
                    axislabel_opts=opts.LabelOpts(margin=30, color="#ffffff63"),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(
                        is_show=True,
                        length=25,
                        linestyle_opts=opts.LineStyleOpts(color="#ffffff1f"),
                    ),
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True, linestyle_opts=opts.LineStyleOpts(color="#ffffff1f")
                    ),
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    position="right",
                    axislabel_opts=opts.LabelOpts(margin=20, color="#ffffff63"),
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(width=2, color="#fff")
                    ),
                    axistick_opts=opts.AxisTickOpts(
                        is_show=True,
                        length=15,
                        linestyle_opts=opts.LineStyleOpts(color="#ffffff1f"),
                    ),
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True, linestyle_opts=opts.LineStyleOpts(color="#ffffff1f")
                    ),
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
        c.render(filename)

    def plot_lines_two(self, x_data, y_data_1, y_data_2, filename):
        background_color_js = (
            "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
            "[{offset: 0, color: '#c86589'}, {offset: 1, color: '#06a7ff'}], false)"
        )
        area_color_js = (
            "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
            "[{offset: 0, color: '#eb64fb'}, {offset: 1, color: '#3fbbff0d'}], false)"
        )
        c = (
            Line(init_opts=opts.InitOpts(bg_color=JsCode(background_color_js)))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
                series_name="l_corner_move",
                y_axis=y_data_1,
                is_smooth=True,
                is_symbol_show=True,
                symbol="circle",
                symbol_size=6,
                label_opts=opts.LabelOpts(is_show=True, position="top", color="white"),
                itemstyle_opts=opts.ItemStyleOpts(
                    color="red", border_color="#fff", border_width=3
                ),
                tooltip_opts=opts.TooltipOpts(is_show=False),
                areastyle_opts=opts.AreaStyleOpts(color=JsCode(area_color_js), opacity=1),
            )
            .add_yaxis(
                series_name="r_corner_move",
                y_axis=y_data_2,
                is_smooth=True,
                is_symbol_show=True,
                symbol="circle",
                symbol_size=6,
                label_opts=opts.LabelOpts(is_show=True, position="top", color="white"),
                itemstyle_opts=opts.ItemStyleOpts(
                    color="red", border_color="#fff", border_width=3
                ),
                tooltip_opts=opts.TooltipOpts(is_show=False),
                areastyle_opts=opts.AreaStyleOpts(color=JsCode(area_color_js), opacity=1),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="show distance between two neighbor frames",
                    pos_top="5%",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(color="#fff", font_size=16),
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    boundary_gap=False,
                    axislabel_opts=opts.LabelOpts(margin=30, color="#ffffff63"),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(
                        is_show=True,
                        length=25,
                        linestyle_opts=opts.LineStyleOpts(color="#ffffff1f"),
                    ),
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True, linestyle_opts=opts.LineStyleOpts(color="#ffffff1f")
                    ),
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    position="right",
                    axislabel_opts=opts.LabelOpts(margin=20, color="#ffffff63"),
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(width=2, color="#fff")
                    ),
                    axistick_opts=opts.AxisTickOpts(
                        is_show=True,
                        length=15,
                        linestyle_opts=opts.LineStyleOpts(color="#ffffff1f"),
                    ),
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True, linestyle_opts=opts.LineStyleOpts(color="#ffffff1f")
                    ),
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
        c.render(filename)

    def show_locate_points(self, image, point):
        tempImage = self.circle_points_on_image(image, point.T)
        cv2.namedWindow('locatePoints', 0)
        cv2.imshow('locatePoints', tempImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()