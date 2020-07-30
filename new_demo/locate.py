# this file is using to locate images

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import time

# this class is for output locate parameters
class OutputLocate(object):
    def __init__(self, params):
        self.params = params
        self.index_list = []
        self.result_list = []

    def save_result(self, index, points_l, points_r, lftpoint, rgtpoint, ptgpoint, kl, kr, kh,
                    bl, br, bh, il, ir, ih):
        self.index_list.append(index)
        result = dict({'index': index, 'points_l': points_l, 'points_r': points_r,
                       'lftpoint': lftpoint, 'rgtpoint': rgtpoint, 'ptgpoint': ptgpoint,
                       'kl': kl, 'kr': kr, 'kh': kh,
                       'bl': bl, 'br': br, 'bh': bh,
                       'il': il, 'ir': ir, 'ih': ih})
        self.result_list.append(result)



class LocateParams(object):
    def __init__(self, template, area, klr_value, kh_value):
        self.area = area
        self.template = template
        self.klr_value = klr_value  # [1, -0.5]
        self.kh_value = kh_value  # [-1, -3.5]


class LocateArea(object):
    def __init__(self, line, pan):
        self.line = line
        self.pan = pan


class LocateTemplate(object):
    def __init__(self, wid_patch_r, wid_patch_c, mean_gray_r, mean_gray_c, dis_r, dis_c):
        self.wid_patch_r, self.wid_patch_c = wid_patch_r, wid_patch_c
        self.mean_gray_r, self.mean_gray_c = mean_gray_r, mean_gray_c
        self.dis_r, self.dis_c = dis_r, dis_c


class Locate(object):
    def __init__(self, params):
        self.params = params
        self.ima_patch_r, self.ima_patch_c = None, None
        self.wid_patch_r, self.wid_patch_c = None, None
        self.get_template()
        self.num_point = 21
        self.b_update = False
        self.output_locate = OutputLocate(params)


    def do_locate_refine(self, img, index, update=0):
        lftpoint = np.zeros([2, self.num_point], dtype='int')
        rgtpoint = np.zeros([2, self.num_point], dtype='int')
        ptgpoint = np.zeros([2, self.num_point], dtype='int')
        start = time.clock()
        if update:
            self.get_template()
        roi_line, roi_pan = self.params.area.line, self.params.area.pan
        mean_gray_r, mean_gray_c = self.params.template.mean_gray_r, self.params.template.mean_gray_c
        img_line = img[roi_line[0]:roi_line[1], roi_line[2]:roi_line[3]]
        img_pan = img[roi_pan[0]:roi_pan[1], roi_pan[2]:roi_pan[3]]
        step = round(img_line.shape[0] / (self.num_point - 1))
        method = cv2.TM_CCORR_NORMED
        # do line template matching for contact wire
        for i in range(self.num_point):
            yi = int(roi_line[0] + i * step)

            img_i = img[int(yi):int(yi + 1), int(roi_line[2]):int(roi_line[3])]
            img_i = img_i - mean_gray_r * np.ones([1, img_line.shape[1]])
            img_i = img_i.astype(np.float32)
            res = cv2.matchTemplate(img_i, self.ima_patch_r, method)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            di = max_loc[0]
            lftpoint[:, i] = [int(di + self.wid_patch_r - 1 + int(roi_line[2]) - 1), yi]
            rgtpoint[:, i] = [int(di + 2 * self.wid_patch_r - 2 + int(roi_line[2]) - 1), yi]

        step = round(img_pan.shape[1] / (self.num_point - 1))
        for i in range(self.num_point):
            xi = int(roi_pan[2] + i * step)
            img_i = img[roi_pan[0]:roi_pan[1], xi:xi + 1].T
            img_i = img_i - mean_gray_c * np.ones([1, img_pan.shape[0]])
            img_i = img_i.astype(np.float32)
            res = cv2.matchTemplate(img_i, self.ima_patch_c, method)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            di = max_loc[0]
            ptgpoint[:, i] = [xi, int(di + roi_pan[0] + self.wid_patch_c - 2)]  # -4是为了保证可以更容易优化到角点

        if lftpoint.shape[1] < 3 or rgtpoint.shape[1] < 3 or ptgpoint.shape[1] < 3:
            kl, kr, kh = 0, 0, 0
            il = [False for i in range(self.num_point)]
            ir, ih = il, il
        else:
            kl, bl, il = ransac(lftpoint, 5)
            kr, br, ir = ransac(rgtpoint, 5)
            kh, bh, ih = ransac(ptgpoint, 2)
            if np.abs(np.mean(lftpoint[0,il])- np.mean(rgtpoint[0,ir])) > 100:
                l_number ,r_number = il.tolist().count(True), ir.tolist().count(True)
                if l_number > r_number > 5 :
                    center = np.mean(lftpoint[0,il])
                    rgtpoint = rgtpoint[:, (rgtpoint[0,:]> center-200) & (rgtpoint[0,:]< center + 200)]
                    kr, br, ir = ransac(rgtpoint, 5)
                elif r_number > l_number > 5:
                    center = np.mean(rgtpoint[0, il])
                    lftpoint = lftpoint[:, (lftpoint[0, :] > center - 200) & (lftpoint[0, :] < center + 200)]
                    kl, bl, il = ransac(lftpoint, 5)
        if kl == 0 or kh == 0:
            xl = np.zeros((1, 2), dtype='float')
            xr = np.zeros((1, 2), dtype='float')
            points = np.vstack((xl, xr))
            self.output_locate.save_result(index, xl, xr, lftpoint, rgtpoint,
                                           ptgpoint, 0, 0, 0, 0, 0, 0, il, ir, ih)
        else:
            xl = np.zeros((1, 2), dtype='float')
            xr = np.zeros((1, 2), dtype='float')
            xli = (bl - bh) / (kh - kl)
            yli = kl * xli + bl
            xri = (br - bh) / (kh - kr)
            yri = kr * xri + br
            xl[:, :] = [xli, yli]
            xr[:, :] = [xri, yri]
            points = do_refine(img, np.vstack((xl, xr)))
            self.output_locate.save_result(index, points[0, :], points[1, :], lftpoint, rgtpoint,
                                           ptgpoint, kl, kr, kh, bl, br, bh, il, ir, ih)
        elapsed = time.clock() - start
        return points, elapsed

    def get_template(self):
        wid_patch_r, wid_patch_c = self.params.template.wid_patch_r, self.params.template.wid_patch_c
        dis_r, dis_c = self.params.template.dis_r, self.params.template.dis_c
        self.ima_patch_r = np.zeros([1, wid_patch_r * 3], dtype='float32')
        self.ima_patch_r[0, wid_patch_r:2 * wid_patch_r] -= np.ones(wid_patch_r) * dis_r / 255
        self.ima_patch_c = np.zeros([1, wid_patch_c * 2], dtype='float32')
        self.ima_patch_c[0, wid_patch_c:2 * wid_patch_c] -= np.ones(wid_patch_c) * dis_c / 255
        self.wid_patch_r, self.wid_patch_c = wid_patch_r, wid_patch_c

    def do_update(self, img, idx):
        locate_output = self.output_locate
        # 无需output时，可将该不放呢修改为，输入一组包含当前locate中间参数的值，替换掉locate_output的输入
        index = locate_output.index_list.index(idx)
        dict_data = locate_output.result_list[index]
        self.b_update = verify_locate(locate_output, idx)
        if self.b_update:
            il = dict_data['il']
            lftpoint = dict_data['lftpoint']
            temp_point = lftpoint[:, il]
            temp_point = temp_point[:, temp_point[1, :] < dict_data['points_l'][1]]
            max_list = []
            min_list = []
            for i in range(temp_point.shape[1]):
                temp_y = temp_point[1, i]
                temp_x = temp_point[0, i] - locate_output.params.template.wid_patch_r
                img_slide = img[temp_y, temp_x:temp_x + locate_output.params.template.wid_patch_r * 3]
                max_list.append(np.max(img_slide))
                min_list.append(np.min(img_slide))
            mean_max_r = np.mean(np.array(max_list))
            mean_min_r = np.mean(np.array(min_list))

            ih = dict_data['ih']
            ptgpoint = dict_data['ptgpoint']
            temp_point = ptgpoint[:, ih]
            temp_point = temp_point[:, (temp_point[0, :] < dict_data['points_l'][0]) &
                                       (temp_point[0, :] < dict_data['points_r'][0])]
            max_list = []
            min_list = []
            for i in range(temp_point.shape[1]):
                temp_x = temp_point[0, i]
                temp_y = temp_point[1, i] - locate_output.params.template.wid_patch_c
                img_slide = img[temp_y: temp_y + locate_output.params.template.wid_patch_c * 2, temp_x]
                max_list.append(np.max(img_slide))
                min_list.append(np.min(img_slide))
            mean_max = np.mean(np.array(max_list))
            mean_min = np.mean(np.array(min_list))
            if np.isnan(mean_max) or np.isnan(mean_min) or np.isnan(mean_max_r) or np.isnan(mean_min_r):
                self.b_update = False
            else:
                self.output_locate.params.template.mean_gray_r = mean_max_r
                self.output_locate.params.template.dis_r = (mean_max_r - mean_min_r) * 256
                self.output_locate.params.template.mean_gray_c = mean_max
                self.output_locate.params.template.dis_c = (mean_max - mean_min)*256
                self.update_params(self.output_locate.params)

    def update_params(self, params):
        self.params = params

def ransac(points, threshold):
    # 进行ransac，用于保障定位的鲁棒性
    ransac_model = RANSACRegressor(LinearRegression(), max_trials=20, min_samples=3,
                                   loss='squared_loss', stop_n_inliers=8,
                                   residual_threshold=threshold, random_state=None)
    line_model = LinearRegression()
    x = points[0:1, :].T
    y = points[1:, :].T
    ransac_model.fit(x, y)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    if inlier_mask.tolist().count(True) < 3:
        d = 0
        k = 0
    else:
        line_model.fit(x[inlier_mask, :], y[inlier_mask, :])

        k_temp = line_model.coef_
        d_temp = line_model.intercept_
        d = ransac_model.predict([[0]])[0, 0]
        k = ransac_model.predict([[1]])[0, 0] - ransac_model.predict([[0]])[0, 0]
    return k, d, inlier_mask


def do_refine(img, points):
    wintx, winty = 9, 9
    points_temp = np.float32(points).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    # img and points must be float32
    corners = cv2.cornerSubPix(img, points_temp, (wintx, winty), (-1, -1), criteria)
    return corners


def verify_locate(locate_output, idx):
    # for track
    index = locate_output.index_list.index(idx)
    dict_data = locate_output.result_list[index]
    # locate
    # dict_data = locate_output.result_list[idx - 1]
    # only for left
    klr_value, kh_value = locate_output.params.klr_value, locate_output.params.kh_value
    # klr_value, kh_value = [-1, -3.5], [1, -0.5]
    if dict_data['kl'] == 0 or dict_data['kr'] == 0 or dict_data['kh'] == 0:
        # print('fitting line error')
        return False
    # only for left
    if klr_value[0] < 2:   # cause left is 1  and right is 3
        if abs(dict_data['kl'] - dict_data['kr']) > 1:
            # print('two fitting line do not match error left ')
            return False

        if dict_data['kl'] > klr_value[0] or dict_data['kl'] < klr_value[1] or \
                dict_data['kr'] > klr_value[0] or dict_data['kr'] < klr_value[1]:
            # print('fitting line k error left')
            return False
    else:  # for right
        if dict_data['kl'] != 0:
            if abs(dict_data['kl'] - dict_data['kr'])/dict_data['kl'] > 0.5:
                # print('two fitting line do not match error right')
                return False
        if not (abs(dict_data['kl']) > klr_value[0] and abs(dict_data['kr']) > klr_value[0]):
            # print('fitting line k error right')
            return False

    if dict_data['kh'] > kh_value[0] or dict_data['kh'] < kh_value[1]:
        # print('fitting line kh error')
        return False
    # only for left
    if not 30 > np.abs(dict_data['points_l'][0] - dict_data['points_r'][0]) > 12:
        # print('points distance error')
        return False
    return True