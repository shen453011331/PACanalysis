# this file is using to save results.
# the results are saved in a csv and contains the idx, process speed, and point results.

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class OutPutResult(object):
    def __init__(self):
        self.result_list = []
        self.index_list = []

    def save_results(self, p_3d, p3d_lhorn, p3d_rhorn, height, width, distance, theta, index):
        self.index_list.append(index)
        result = dict({'index': index, 'points_3d': p_3d, 'p3d_lhorn': p3d_lhorn,
                       'p3d_rhorn': p3d_rhorn, 'height': height, 'width': width,
                       'distance': distance, 'theta': theta})
        self.result_list.append(result)


class SaveResult(object):

    def __init__(self, path):
        self.path = path
        self.contact_volumn = ['number',
                               'l_p_l_lo_x', 'l_p_l_lo_y', 'l_p_l_re_x', 'l_p_l_re_y',
                               'l_p_r_lo_x', 'l_p_r_lo_y', 'l_p_r_re_x', 'l_p_r_re_y']
        self.df_contact = pd.DataFrame(columns=self.contact_volumn)


    def save_values(self, index, points_l_lft, points_lft):
        # save one image contact points
        points_l_lft = np.round(points_l_lft, decimals=2)
        points_lft = np.round(points_lft, decimals=2)
        temp_data = {'number': index,
                     'l_p_l_lo_x': points_l_lft[0, 0], 'l_p_l_lo_y': points_l_lft[0, 1],
                     'l_p_l_re_x': points_lft[0, 0], 'l_p_l_re_y': points_lft[0, 1],
                     'l_p_r_lo_x': points_l_lft[1, 0], 'l_p_r_lo_y': points_l_lft[1, 1],
                     'l_p_r_re_x': points_lft[1, 0], 'l_p_r_re_y': points_lft[1, 1],
                     }
        temp_df = pd.DataFrame(temp_data, index=[0])
        self.df_contact = self.df_contact.append(temp_df)
        return self.df_contact





