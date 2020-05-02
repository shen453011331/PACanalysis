# 2020.1.2 this is file to analysis the input data
import numpy as np
import talib as ta
import pandas as pd

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

