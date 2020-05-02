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
        self.plot_line_path = ''
        pass

    def plot_lines(self, x_data, y_data, filename_, series_name='x1_dist'):
        filename = self.plot_line_path + filename_
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
                series_name=series_name,
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
                    title="show %s in one image" % series_name,
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

    def circle_points_on_image(self, image, points, color=(255, 255, 0)):
        # points are N*2 [x,y] numpy
        if len(points.shape) == 1:
            cv2.circle(image, (int(points[0]), int(points[1])), 5, color, 2)
        else:
            for i in range(points.shape[0]):
                cv2.circle(image, (int(points[i, 0]), int(points[i, 1])), 5, color, 2)
        return image

    def draw_line_on_image(self, image, k, b, color=(255, 255, 0)):
        y0, y1 = 0, 500
        if k != 0:
            x0 = int((y0 - b)/k)
            x1 = int((y1 - b)/k)
            cv2.line(image, (x0, y0), (x1, y1), color, 3)
        return image

    def draw_locate(self, img, locate_output, idx):
        dict_data = locate_output.result_list[idx - 1]
        assert dict_data['index'] == idx, 'old {}, new {}'.format(dict_data['index'], idx)
        img = self.draw_line_on_image(img, dict_data['kl'], dict_data['bl'], (255, 0, 0))
        img = self.draw_line_on_image(img, dict_data['kr'], dict_data['br'], (255, 0, 0))
        img = self.draw_line_on_image(img, dict_data['kh'], dict_data['bh'], (255, 0, 0))
        lftpoint = dict_data['lftpoint']
        rgtpoint = dict_data['rgtpoint']
        ptgpoint = dict_data['ptgpoint']
        img = self.circle_points_on_image(img, lftpoint.T)
        img = self.circle_points_on_image(img, rgtpoint.T)
        img = self.circle_points_on_image(img, ptgpoint.T)
        img = self.circle_points_on_image(img, lftpoint[:, dict_data['il']].T, (255, 0, 0))
        img = self.circle_points_on_image(img, rgtpoint[:, dict_data['ir']].T, (255, 0, 0))
        img = self.circle_points_on_image(img, ptgpoint[:, dict_data['ih']].T, (255, 0, 0))
        img = self.circle_points_on_image(img, dict_data['points_l'], (0, 0, 255))
        img = self.circle_points_on_image(img, dict_data['points_r'], (0, 0, 255))
        return img
