from plotresult import PlotResult
import pyecharts.options as opts
from pyecharts.charts import Line, Page, Scatter
from pyecharts.commons.utils import JsCode
from pyecharts.faker import Collector, Faker
import numpy as np
import math


def get_ith_points(pan_points_3d, p3d_lhorn, theta, height, length, pan_length_tgt, pan_height_tgt, idx):
    if 0 in p3d_lhorn:
        return None
    rotate_matrix = np.array([[math.cos(theta), math.sin(theta)],
                              [-math.sin(theta), math.cos(theta)]])
    temp_pan_3d = pan_points_3d.copy()
    # time sequence
    temp_pan_3d[:, 0] = temp_pan_3d[:, 0] + idx
    # zoom
    temp_pan_3d[:, 1] = temp_pan_3d[:, 1] * length / pan_length_tgt
    temp_pan_3d[:, 2] = temp_pan_3d[:, 2] * height / pan_height_tgt
    # rotate
    temp_pan_3d[:, 1:] = np.dot(rotate_matrix, pan_points_3d[:, 1:].T).T
    # shift
    temp_pan_3d[:, 0] = temp_pan_3d[:, 0] + np.squeeze(p3d_lhorn)[1]
    temp_pan_3d[:, 1] = temp_pan_3d[:, 1] + np.squeeze(p3d_lhorn)[0]
    temp_pan_3d[:, 2] = temp_pan_3d[:, 2] + np.squeeze(p3d_lhorn)[2]

    return temp_pan_3d


def get_color(scalar, color_sample):
    i1, i2 = int(scalar * 64), int(scalar * 64) + 1
    if i1 == 0:
        return color_sample[0]
    alpha = (scalar*64-i1)
    color = color_sample[i1-1] * (1-alpha) + color_sample[i2-1] * alpha
    return color


def get_images(x_data, y_data, filename_, series_name='x1_dist'):
    plot_line_path = '.'
    filename = plot_line_path + filename_
    background_color_js = (
        "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
        "[{offset: 0, color: '#c86589'}, {offset: 1, color: '#06a7ff'}], false)"
    )
    area_color_js = (
        "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
        "[{offset: 0, color: '#eb64fb'}, {offset: 1, color: '#3fbbff0d'}], false)"
    )
    c = (
        Scatter(init_opts=opts.InitOpts(bg_color=JsCode(background_color_js)))
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
                min_='dataMin',
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