# this file is using to save results.
# the results are saved in a csv and contains the idx, process speed, and point results.

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



class SaveResult(object):

    def __init__(self, path):
        self.path = path

        self.df = pd.DataFrame(columns=['number', 'point_l_x', 'point_l_y', 'point_2_x', 'point_2_y', 'speed'])
        self.temp_df = None

        self.temp_column =['dist1_l', 'dist1_r',
                           'dist2_l_l','dist2_l_r','dist2_r_l','dist2_r_r',
                           'dist3_l', 'dist3_r' ,'speed_err_l', 'speed_err_r' ,
                           'dist4','number','shift1']
        self.temp_data = {}
        for i in self.temp_column:
            self.temp_data[i] = 0
        self.all_value_column = ['number',
                                 'l_p_l_lo_x', 'l_p_l_lo_y', 'l_p_l_re_x', 'l_p_l_re_y',
                                 'l_p_r_lo_x', 'l_p_r_lo_y', 'l_p_r_re_x', 'l_p_r_re_y',
                                 'l_h_l_x', 'l_h_l_y', 'l_h_r_x', 'l_h_r_y',
                                 'r_p_l_lo_x', 'r_p_l_lo_y', 'r_p_l_re_x', 'r_p_l_re_y',
                                 'r_p_r_lo_x', 'r_p_r_lo_y', 'r_p_r_re_x', 'r_p_r_re_y',
                                 'r_h_l_x', 'r_h_l_y', 'r_h_r_x', 'r_h_r_y',
                                 '3d_p_x', '3d_p_y', '3d_p_z',
                                 '3d_h_l_x', '3d_h_l_y', '3d_h_l_z',
                                 '3d_h_r_x', '3d_h_r_y', '3d_h_r_z']
        self.df_all_value = pd.DataFrame(columns = self.all_value_column)

    def saveResult(self, points, idx, speed, filename):
        tgt_name = self.path + filename
        temp_data = {
            'number': idx,
            'point_l_x': points[0, 0],
            'point_l_y': points[0, 1],
            'point_2_x': points[1, 0],
            'point_2_y': points[1, 1],
            'speed': speed
        }
        self.temp_df = pd.DataFrame(temp_data, index=[0])
        if os.path.exists(tgt_name):
            self.temp_df.to_csv(tgt_name, mode='a', header=False)
            pass
        else:
            self.df = self.df.append(self.temp_df)
            self.df.to_csv(tgt_name)
        pass

    def plot_csv(self, csv_file):

        for i in csv_file:
            plt.figure()
            df = pd.read_csv(self.path + i)
            y = df['point_l_x'].values
            mask1 = y > 0
            mask2 = y < 2336
            mask = mask1 & mask2
            y0 = y[mask]
            x = np.linspace(1, len(y), len(y))
            x0 = x[mask]
            plt.scatter(x0, y0, s=3, c='red', alpha=0.5)
            plt.show()

    def save3DResult(self, kind, P3D):
        self.addResultKind(kind + '_x', P3D[0, 0])
        self.addResultKind(kind + '_y', P3D[1, 0])
        self.addResultKind(kind + '_z', P3D[2, 0])
        pass

    def addResultKind(self, kind, value, init=0):
        self.temp_data[kind] = value

    def saveMutiKindResult(self, idx, filename):
        self.temp_data['number'] = idx
        tgt_name = self.path + 'run_data/' + filename
        self.temp_df = pd.DataFrame(self.temp_data, index=[0])
        if os.path.exists(tgt_name):
            self.temp_df.to_csv(tgt_name, mode='a', header=False, columns=self.temp_column)
            pass
        else:
            df = pd.DataFrame(columns=self.temp_column)
            df = df.append(self.temp_df)
            df.to_csv(tgt_name, columns=self.temp_column)


    def save_all_value(self, index, points_l_lft, points_l_rgt, points_lft, points_rgt,
                       lhorn_l, lhorn_r, rhorn_l, rhorn_r, P, P_lhorn, P_rhorn, result_value_file):
        # save all raw data
        # format example l_p_l_lo_x
        # format as  l/r/3d _  data from left or right image or 3d
        # p_/h_ is contact points or horn points
        # l_/r_ is left or right points in the frame
        # _lo/_re is point result with refine or without   only for contact points
        # _x/_y/_z is value of which coordinate
        points_l_lft = np.round(points_l_lft, decimals=2)
        points_l_rgt = np.round(points_l_rgt, decimals=2)
        points_lft = np.round(points_lft, decimals=2)
        points_rgt = np.round(points_rgt, decimals=2)
        lhorn_l = np.round(lhorn_l, decimals=2)
        lhorn_r = np.round(lhorn_r, decimals=2)
        rhorn_l = np.round(rhorn_l, decimals=2)
        rhorn_r = np.round(rhorn_r, decimals=2)
        P = np.round(P, decimals=2)
        P_lhorn = np.round(P_lhorn, decimals=2)
        P_rhorn = np.round(P_rhorn, decimals=2)
        temp_data = {'number': index,
                     'l_p_l_lo_x': points_l_lft[0, 0], 'l_p_l_lo_y': points_l_lft[0, 1],
                     'l_p_l_re_x': points_lft[0, 0], 'l_p_l_re_y': points_lft[0, 1],
                     'l_p_r_lo_x': points_l_lft[1, 0], 'l_p_r_lo_y': points_l_lft[1, 1],
                     'l_p_r_re_x': points_lft[1, 0], 'l_p_r_re_y': points_lft[1, 1],
                     'l_h_l_x': lhorn_l[0], 'l_h_l_y': lhorn_l[1],
                     'l_h_r_x': lhorn_r[0], 'l_h_r_y': lhorn_r[1],
                     'r_p_l_lo_x': points_l_rgt[0, 0], 'r_p_l_lo_y': points_l_rgt[0, 1],
                     'r_p_l_re_x': points_rgt[0, 0], 'r_p_l_re_y': points_rgt[0, 1],
                     'r_p_r_lo_x': points_l_rgt[1, 0], 'r_p_r_lo_y': points_l_rgt[1, 1],
                     'r_p_r_re_x': points_rgt[1, 0], 'r_p_r_re_y': points_rgt[1, 1],
                     'r_h_l_x': rhorn_l[0], 'r_h_l_y': rhorn_l[1],
                     'r_h_r_x': rhorn_r[0], 'r_h_r_y': rhorn_r[1],
                     '3d_p_x': P[0, 0], '3d_p_y': P[1, 0], '3d_p_z': P[2, 0],
                     '3d_h_l_x': P_lhorn[0, 0], '3d_h_l_y': P_lhorn[1, 0], '3d_h_l_z': P_lhorn[2, 0],
                     '3d_h_r_x': P_rhorn[0, 0], '3d_h_r_y': P_rhorn[1, 0], '3d_h_r_z': P_rhorn[2, 0]
                     }
        temp_df = pd.DataFrame(temp_data, index=[0])
        tgt_name = self.path + 'run_data/' + result_value_file
        self.df_all_value = self.df_all_value.append(temp_df)
        if os.path.exists(tgt_name):
            temp_df.to_csv(tgt_name, mode='a', header=False, columns=self.all_value_column)
        else:
            self.df_all_value.to_csv(tgt_name, columns=self.all_value_column)








