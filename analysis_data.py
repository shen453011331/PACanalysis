# 2020.1.2 this is file to analysis the input data
import numpy as np
import talib as ta

class AnalysisData(object):
    def __init__(self, save_process, pro3d, plot):
        self.save = save_process
        self.pro3d = pro3d
        self.plot = plot
        pass

    def analysis_all(self, df, index):
        df_now = df[df['number'] == index]
        self.analysis_dist1(df_now)
        self.analysis_dist2(df_now)

        df_seq = df[df['number'] <= index]
        if len(df_seq) > 1:
            df_last = df[df['number'] == index-1]
            self.analysis_dist3(df_now, df_last)
        if len(df_seq) > 5:
            self.analysis_speed_error(df_seq)
        P_C, P_L, P_R = self.get_3D_from_df(df_now)
        shift1, dist4 = self.pro3d.calculateShift(P_C, P_L, P_R)
        self.save.addResultKind('shift1', shift1)
        self.save.addResultKind('dist4', dist4)
        self.save.saveMutiKindResult(index, 'analysis.csv')

    def analysis_dist1(self, df_now):
        l_p_l_re, l_p_r_re, l_p_l_lo, l_p_r_lo = self.get_left_point_from_df(df_now)
        r_p_l_re, r_p_r_re, r_p_l_lo, r_p_r_lo = self.get_right_point_from_df(df_now)
        dist1_l = self.get_dist1(l_p_l_re, l_p_r_re)
        dist1_r = self.get_dist1(r_p_l_re, r_p_r_re)
        # print('left image:two contact points distance is %.2f' % dist1)
        self.save.addResultKind('dist1_l', dist1_l)
        self.save.addResultKind('dist1_r', dist1_r)
        return np.array([dist1_l, dist1_r])

    def analysis_dist2(self, df_now):
        l_p_l_re, l_p_r_re, l_p_l_lo, l_p_r_lo = self.get_left_point_from_df(df_now)
        r_p_l_re, r_p_r_re, r_p_l_lo, r_p_r_lo = self.get_right_point_from_df(df_now)
        # dist_l_r  dist in left image for the right points
        dist2_l_l = self.get_dist2(l_p_l_lo, l_p_l_re)
        dist2_l_r = self.get_dist2(l_p_r_lo, l_p_r_re)
        dist2_r_l = self.get_dist2(r_p_l_lo, r_p_l_re)
        dist2_r_r = self.get_dist2(r_p_r_lo, r_p_r_re)
        # print('left image:left contact points refine distance is %.2f' % dist2)
        self.save.addResultKind('dist2_l_l', dist2_l_l)
        self.save.addResultKind('dist2_l_r', dist2_l_r)
        self.save.addResultKind('dist2_r_l', dist2_r_l)
        self.save.addResultKind('dist2_r_r', dist2_r_r)
        return np.array([dist2_l_l, dist2_l_r, dist2_r_l, dist2_r_r])

    def analysis_dist3(self, df_now, df_last):
        l_p_l_re, l_p_r_re, l_p_l_lo, l_p_r_lo = self.get_left_point_from_df(df_now)
        r_p_l_re, r_p_r_re, r_p_l_lo, r_p_r_lo = self.get_right_point_from_df(df_now)
        l_p = np.mean(np.vstack([l_p_l_re, l_p_r_re]), axis=0)
        r_p = np.mean(np.vstack([r_p_l_re, r_p_r_re]), axis=0)
        l_p_l_re_2, l_p_r_re_2, _, _ = self.get_left_point_from_df(df_last)
        r_p_l_re_2, r_p_r_re_2, _, _ = self.get_right_point_from_df(df_last)
        l_p_2 = np.mean(np.vstack([l_p_l_re_2, l_p_r_re_2]), axis=0)
        r_p_2 = np.mean(np.vstack([r_p_l_re_2, r_p_r_re_2]), axis=0)
        dist3_l = self.get_dist3(l_p, l_p_2)
        dist3_r = self.get_dist3(r_p, r_p_2)
        self.save.addResultKind('dist3_l', dist3_l)
        self.save.addResultKind('dist3_r', dist3_r)
        return np.array([dist3_l, dist3_r])

    def analysis_speed_error(self, df_seq):
        l_p_l_re_seq, l_p_r_re_seq, _, _ = self.get_left_point_from_df(df_seq)
        r_p_l_re_seq, r_p_r_re_seq, _, _ = self.get_right_point_from_df(df_seq)
        l_p_x = np.mean(np.hstack([l_p_l_re_seq[:, 0:1], l_p_r_re_seq[:, 0:1]]), axis=1)
        l_p_y = np.mean(np.hstack([l_p_l_re_seq[:, 1:], l_p_r_re_seq[:, 1:]]), axis=1)
        l_p_seq = np.vstack([l_p_x, l_p_y]).T
        r_p_x = np.mean(np.hstack([r_p_l_re_seq[:, 0:1], r_p_r_re_seq[:, 0:1]]), axis=1)
        r_p_y = np.mean(np.hstack([r_p_l_re_seq[:, 1:], r_p_r_re_seq[:, 1:]]), axis=1)
        r_p_seq = np.vstack([r_p_x, r_p_y]).T
        speed_err_l = self.get_speed_error(l_p_seq)
        speed_err_r = self.get_speed_error(r_p_seq)
        self.save.addResultKind('speed_err_l', speed_err_l)
        self.save.addResultKind('speed_err_r', speed_err_r)
        # print('left image:left contact points move error is %.2f' % speed_err)

    def get_left_point_from_df(self, df):
        l_p_l_re = df[['l_p_l_re_x', 'l_p_l_re_y']].values
        l_p_r_re = df[['l_p_r_re_x', 'l_p_r_re_y']].values
        l_p_l_lo = df[['l_p_l_lo_x', 'l_p_l_lo_y']].values
        l_p_r_lo = df[['l_p_r_lo_x', 'l_p_r_lo_y']].values
        return l_p_l_re, l_p_r_re, l_p_l_lo, l_p_r_lo

    def get_right_point_from_df(self, df):
        r_p_l_re = df[['r_p_l_re_x', 'r_p_l_re_y']].values
        r_p_r_re = df[['r_p_r_re_x', 'r_p_r_re_y']].values
        r_p_l_lo = df[['r_p_l_lo_x', 'r_p_l_lo_y']].values
        r_p_r_lo = df[['r_p_r_lo_x', 'r_p_r_lo_y']].values
        return r_p_l_re, r_p_r_re, r_p_l_lo, r_p_r_lo

    def get_3D_from_df(self, df):
        P_C = df[['3d_p_x', '3d_p_y', '3d_p_z']].values
        P_L = df[['3d_h_l_x', '3d_h_l_y', '3d_h_l_z']].values
        P_R = df[['3d_h_r_x', '3d_h_r_y', '3d_h_r_z']].values
        return P_C, P_L, P_R

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

