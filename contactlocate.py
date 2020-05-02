# this file is using to locate the contact points
# output points [x-l, y-l ; x-r, y-r]
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import time

imageWidth = 2336
imageHeight = 900


class ContactLocate(object):

    def __init__(self, locate_params, track_params, refine_params, verify_params):
        self.locate_params = locate_params
        self.track_params = track_params
        self.startNum = 1

        self.meangrayr = self.locate_params['meangrayr']
        self.meangrayc = self.locate_params['meangrayc']
        self.widpatchr = self.locate_params['widpatchr']
        self.widpatchc = self.locate_params['widpatchc']
        self.disr = self.locate_params['disr']
        self.disc = self.locate_params['disc']
        self.imapatchr = None
        self.imapatchc = None
        self.numpoint = 21
        self.lftpoint = np.zeros([2, self.numpoint], dtype='int')
        self.rgtpoint = np.zeros([2, self.numpoint], dtype='int')
        self.ptgpoint = np.zeros([2, self.numpoint], dtype='int')
        self.kl = 0
        self.kr = 0
        self.kh = 0
        self.LineROI = None
        self.PanROI = None

        # track
        self.tracker = cv2.TrackerKCF_create()
        self.wintx = 9
        self.winty = 9
        self.track_length = 0  # 控制跟踪序列init更新间距
        self.localtracker_l = cv2.TrackerKCF_create()
        self.localtracker_r = cv2.TrackerKCF_create()
        self.widthx = 100
        self.heighty = 50

        # verify
        self.points_dist = [17, 22]
        self.thre_2 = 5

        # bool
        self.b_locate = False
        self.b_muti_track = False
        self.b_init = False
        self.bbox = None
        self.bbox_local_r = None
        self.bbox_local_l = None

        # global
        self.show = None
        self.save = None
        self.ana = None

        # sequence
        self.old_image = None
        self.old_points = None

        # points
        self.p_lo = None # points after locate
        self.p_re = None # points after refine

        # time
        self.sum_time = 0
        self.sum_num = 0
        self.fps = 0
        self.locate_num = 0


    def corner_locating_seprete(self, image, fImage, index):
        self.image = image
        self.index = index
        start = time.clock()
        p_lo, p_re = self.do_position(image, fImage, index, only_locate=1)
        elapsed = time.clock() - start
        self.do_analysis(image, fImage, index)
        self.do_update(image)
        # get fps
        self.sum_time = self.sum_time + elapsed
        self.sum_num = self.sum_num + 1
        self.fps = self.sum_time / self.sum_num
        return p_re, self.fps

    def do_position(self, image, fImage, img_idx, only_locate=0, only_track=0):
        if only_locate:
            p_lo = self.do_locate(fImage)
        else:
            if img_idx == self.startNum or self.b_locate:  # start
                print('do locate')
                track_length = 0
                p_lo = self.do_locate(fImage)
            else:
                print('do track')
                if only_track:
                    p_lo = self.do_track(self.old_image, image, self.old_points)
                else:
                    if self.b_muti_track:
                        p_lo = self.do_local_track(self.old_image, image, self.old_points)
                    else:
                        p_lo = self.do_only_sub_track(self.old_image, image, self.old_points)
                self.track_length = self.track_length + 1
        self.p_re = self.do_refine(fImage, p_lo)
        self.p_lo = p_lo
        return self.p_lo, self.p_re

    def do_analysis(self, image, fImage, index):
        df_contact = self.save.save_values(index, self.p_lo, self.p_re)
        dist1, dist2_l, dist2_r, dist3_l, speed_err_l = self.ana.analysis_contact(df_contact, index)

        if self.b_locate:
            # 当前帧是通过检测定位得到，则下一帧一旦跟踪，需要重新初始化跟踪框
            self.b_init = False
            if self.points_dist[0] < dist1 < self.points_dist[1]:
                if max([dist2_r, dist2_l]) > self.thre_2:
                    # 针对的是以为检测定位成功，实际左右优化后结果偏差太大，这时应该选择之前的直线交错点结果
                    # 并且下一帧重新跟踪
                    print('signal 1')
                    self.p_re = self.p_lo
                    self.b_locate = True
                else:
                    # 定位成功，下一帧执行跟踪
                    self.b_locate = False
                    self.b_muti_track = True
                    print('signal 2')
            else:
                # 检测定位失败
                self.p_re = self.p_lo
                self.b_locate = True
                print('signal 3')
        else:
            # 当前帧为跟踪
            if self.points_dist[0] < dist1 < self.points_dist[1] \
                    and max([dist2_r, dist2_l]) < self.thre_2 \
                    and min([dist2_r, dist2_l]) > 0:
                # 跟踪结果很好
                print('signal 4')
                self.b_muti_track = True
                if self.track_length > 5 and dist3_l < 10 and speed_err_l < 1:
                    # 若运动轨迹非常稳定，则只进行局部跟踪
                    self.b_muti_track = False
                    print('signal 5')
            else:
                # 跟踪失败，下一帧重新检测，当前帧使用检测结果
                self.b_locate = True
                p_lo, p_re = self.do_position(image, fImage, index)
                print('signal 6')
        pass

    def do_update(self, image):
        self.old_image = image
        self.old_points = self.p_re

    def do_locate(self, fImage):
        self.locate_num = self.locate_num + 1
        self.LineROI = [0, 400, 600, 1900]  # 2100
        self.PanROI = [300, 550, 800, 1600]  # 300 550
        self.updateTemplate()
        self.locateImage(self.LineROI, self.PanROI, fImage)
        locatePoints = self.doCrossingPoints(self.lftpoint, self.rgtpoint, self.ptgpoint)
        self.save_locate_images(locatePoints)
        return locatePoints

    def do_track(self, oldImage, image, points, resize=0):
        self.getRectFromPoints(points)
        old_bbox = self.bbox
        if not self.b_init:
            self.tracker = None
            self.tracker = cv2.TrackerKCF_create()
            ok = self.tracker.init(oldImage, self.bbox)
            ok, self.bbox = self.tracker.update(oldImage)
            self.b_init = True
        ok, self.bbox = self.tracker.update(image)
        self.getPointsFromRect()

        # filename = 'result_images/locateImages/%06d.png' % self.index
        # saveImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        # # plot LineROi, PanROI
        # cv2.rectangle(saveImage, (old_bbox[2], self.LineROI[0]),
        #               (self.LineROI[3], self.LineROI[1]), (255, 0, 0))
        # cv2.rectangle(saveImage, (self.PanROI[2], self.PanROI[0]),
        #               (self.PanROI[3], self.PanROI[1]), (0, 255, 0))
        # saveImage = self.show.circle_points_on_image(saveImage, self.lftpoint.T)
        # saveImage = self.show.circle_points_on_image(saveImage, self.rgtpoint.T)
        return self.trackPoints

    def do_local_track(self, oldImage, image, points_l):
        self.getRectFromPoints(points_l)
        old_box = self.bbox
        old_patch = self.get_patch(oldImage, self.bbox)
        if not self.b_init:
            self.tracker = None
            self.tracker = cv2.TrackerKCF_create()
            ok = self.tracker.init(oldImage, self.bbox)
            ok, self.bbox = self.tracker.update(oldImage)
            self.b_init_l = False
            self.b_init_r = False
            self.b_init = True
        ok, self.bbox = self.tracker.update(image)
        new_box = self.bbox
        new_patch = self.get_patch(image, self.bbox)
        self.trackPoints = points_l  # just for giving a shape for trackpoints
        self.GloalTrackPoints = self.getPointsFromRect()
        # do subTracking
        self.do_sub_track(points_l, old_box, new_box, self.localtracker_l, old_patch,
                     new_patch, self.bbox_local_l, 0, self.b_init_l)
        self.do_sub_track(points_l, old_box, new_box, self.localtracker_r, old_patch,
                          new_patch, self.bbox_local_r, 1, self.b_init_r)
        pass
        return self.trackPoints

    def do_only_sub_track(self, oldImage, image, points_l):
        self.getRectFromPoints(points_l)
        old_box = self.bbox
        old_patch = self.get_patch(oldImage, self.bbox)
        new_box = old_box
        new_patch = self.get_patch(image, self.bbox)
        self.trackPoints = points_l  # just for giving a shape for trackpoints
        # do subTracking
        self.do_sub_track(points_l, old_box, new_box, self.localtracker_l, old_patch,
                     new_patch, self.bbox_local_l, 0, self.b_init_l)
        self.do_sub_track(points_l, old_box, new_box, self.localtracker_r, old_patch,
                          new_patch, self.bbox_local_r, 1, self.b_init_r)
        pass
        return self.trackPoints

    def do_refine(self, fImage, points):
        points_temp = np.float32(points).copy()
        floatI = np.float32(fImage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(floatI, points_temp, (self.wintx, self.winty), (-1, -1), criteria)
        self.refinePoints = corners
        return self.refinePoints

    def updateTemplate(self):
        # 为后续模板参数更新做准备，将模板根据参数重新调整
        self.imapatchr = np.zeros([1, self.widpatchr * 3], dtype='float32')
        self.imapatchr[0, self.widpatchr:2 * self.widpatchr] -= np.ones(self.widpatchr) * self.disr / 255
        self.imapatchc = np.zeros([1, self.widpatchc * 2], dtype='float32')
        self.imapatchc[0, self.widpatchc:2 * self.widpatchc] -= np.ones(self.widpatchc) * self.disc / 255

    def locateImage(self, roi_line, roi_pan, fImage):
        # 根据的接触线和受电弓区域进行模板匹配，这里区域都为矩形，且一旦相机位置不变，该区域不变

        self.I_Line = fImage[roi_line[0]:roi_line[1], roi_line[2]:roi_line[3]]
        self.I_Pan = fImage[roi_pan[0]:roi_pan[1], roi_pan[2]:roi_pan[3]]
        step = round(self.I_Line.shape[0] / (self.numpoint - 1))
        method = cv2.TM_CCORR_NORMED

        # do line template matching for contact wire
        for i in range(self.numpoint):
            yi = int(roi_line[0] + i * step)

            Li = fImage[int(yi):int(yi + 1), int(roi_line[2]):int(roi_line[3])]
            Li = Li - self.meangrayr * np.ones([1, self.I_Line.shape[1]])
            Li = Li.astype(np.float32)
            res = cv2.matchTemplate(Li, self.imapatchr, method)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            di = max_loc[0]
            self.lftpoint[:, i] = [int(di + self.widpatchr - 1 + int(roi_line[2]) - 1), yi]
            self.rgtpoint[:, i] = [int(di + 2 * self.widpatchr - 2 + int(roi_line[2]) - 1), yi]

        step = round(self.I_Pan.shape[1] / (self.numpoint - 1))
        for i in range(self.numpoint):
            xi = int(roi_pan[2] + i * step)
            Li = fImage[roi_pan[0]:roi_pan[1], xi:xi + 1].T
            Li = Li - self.meangrayc * np.ones([1, self.I_Pan.shape[0]])
            Li = Li.astype(np.float32)
            res = cv2.matchTemplate(Li, self.imapatchc, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            di = max_loc[0]
            self.ptgpoint[:, i] = [xi, int(di + roi_pan[0] + self.widpatchc - 2)]  # -4是为了保证可以更容易优化到角点

    def doCrossingPoints(self, lftpoint, rgtpoint, ptgpoint):
        # 根据模板匹配获取的点集进行直线交错
        if lftpoint.shape[1] < 3 or lftpoint.shape[1] < 3 or lftpoint.shape[1] < 3:
            self.kl = 0
            self.kr = 0
            self.kh = 0
        else:
            self.kl, bl, self.il = ransac(lftpoint, 5)
            self.kr, br, self.ir = ransac(rgtpoint, 5)
            self.kh, bh, self.ih = ransac(ptgpoint, 2)
        if self.kl == 0 or self.kh == 0:
            xl = np.zeros((1, 2), dtype='float')
            xr = np.zeros((1, 2), dtype='float')
        else:
            xl = np.zeros((1, 2), dtype='float')
            xr = np.zeros((1, 2), dtype='float')
            # self.showPoints(lftpoint[:,il])
            xli = (bl - bh) / (self.kh - self.kl)
            yli = self.kl * xli + bl
            xri = (br - bh) / (self.kh - self.kr)
            yri = self.kr * xri + br
            xl[:, :] = [xli, yli]
            xr[:, :] = [xri, yri]
        return np.vstack((xl, xr))

    def getRectFromPoints(self, points):
        widthx = self.widthx
        heighty = self.heighty
        left = int(np.mean(points[:, 0]) - widthx / 2)
        up = int(np.mean(points[:, 1]) - heighty / 2)
        shiftx = points[1, 0] - np.mean(points[:, 0])
        shifty = points[1, 1] - np.mean(points[:, 1])
        self.shift = (shiftx, shifty)
        self.bbox = (left, up, widthx, heighty)

    def getPointsFromRect(self):
        bbox = self.bbox
        midPoints = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))

        midPoints = np.array(midPoints, dtype=float)
        shift = np.array(self.shift, dtype=float)
        self.trackPoints = np.zeros([2, 2], dtype='float')
        self.trackPoints[0, :] = midPoints - shift
        self.trackPoints[1, :] = midPoints + shift
        return self.trackPoints

    def get_patch(self, image, bbox):
        x1 = int(bbox[0])
        x2 = int(bbox[0] + bbox[2])
        y1 = int(bbox[1])
        y2 = int(bbox[1] + bbox[3])
        return image[y1:y2, x1:x2]

    def do_sub_track(self, points, old_box, new_box, sub_tracker, old_patch,
                     new_patch, bbox_local, b_rl, b_init):
        # b_rl: 0 is left, 1 is right
        local_bbox = self.getLocalRect(points[b_rl, :], old_box)
        if not b_init:
            sub_tracker = None
            sub_tracker = cv2.TrackerKCF_create()
            ok = sub_tracker.init(old_patch, local_bbox)
            ok, bbox_local = sub_tracker.update(old_patch)
            b_init = True
        ok, bbox_local = sub_tracker.update(new_patch)
        if ok:
            self.trackPoints[b_rl, 0] = bbox_local[0] + bbox_local[2] / 2 + self.bbox[0]
            self.trackPoints[b_rl, 1] = bbox_local[1] + bbox_local[3] / 2 + self.bbox[1]
        else:
            bbox = self.bbox
            midPoints = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            midPoints = np.array(midPoints, dtype=float)
            shift = np.array(self.shift, dtype=float)
            if not ok:
                if not b_rl:
                    self.trackPoints[0, :] = midPoints - shift
                else:
                    self.trackPoints[1, :] = midPoints + shift

    def getLocalRect(self, points, bbox):
        widthx = 30
        heighty = 30
        left = int(points[0] - widthx / 2) - bbox[0]
        up = int(points[1] - heighty / 2) - bbox[1]
        return (left, up, widthx, heighty)

    def corner_locating_seprete_only_locate(self, image, fImage, index):
        start = time.clock()
        p_lo, p_re = self.do_position(image, fImage, index, only_locate=1)
        elapsed = time.clock() - start
        # get fps
        self.sum_time = self.sum_time + elapsed
        self.sum_num = self.sum_num + 1
        self.fps = self.sum_time / self.sum_num
        return p_re, self.fps

    def corner_locating_seprete_only_track(self, image, fImage, index):
        start = time.clock()
        p_lo, p_re = self.do_position(image, fImage, index, only_track=1)
        elapsed = time.clock() - start
        self.do_analysis(image, fImage, index)
        self.do_update(image)
        # get fps
        self.sum_time = self.sum_time + elapsed
        self.sum_num = self.sum_num + 1
        self.fps = self.sum_time / self.sum_num
        return p_re, self.fps

    def save_locate_images(self, locatePoints):
        # save locate file
        filename = 'result_images/locateImages/%06d.png' % self.index
        saveImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        # plot LineROi, PanROI
        cv2.rectangle(saveImage, (self.LineROI[2], self.LineROI[0]),
                      (self.LineROI[3], self.LineROI[1]), (255, 0, 0))
        cv2.rectangle(saveImage, (self.PanROI[2], self.PanROI[0]),
                      (self.PanROI[3], self.PanROI[1]), (0, 255, 0))
        saveImage = self.show.circle_points_on_image(saveImage, self.lftpoint.T)
        saveImage = self.show.circle_points_on_image(saveImage, self.rgtpoint.T)
        saveImage = self.show.circle_points_on_image(saveImage, self.ptgpoint.T)
        saveImage = self.show.circle_points_on_image(saveImage, locatePoints, (0, 0, 255))
        cv2.imwrite(filename, saveImage)



def ransac(points, threshold):
    # 进行ransac，用于保障定位的鲁棒性
    ransac_model = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=3,
                                   loss='squared_loss', stop_n_inliers=8,
                                   residual_threshold=threshold, random_state=None)
    X = points[0:1, :].T
    Y = points[1:, :].T

    ransac_model.fit(X, Y)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    if len(inlier_mask) < 3:
        d = 0
        k = 0
    else:
        d = ransac_model.predict([[0]])[0, 0]
        k = ransac_model.predict([[1]])[0, 0] - ransac_model.predict([[0]])[0, 0]
    return k, d, inlier_mask


def pHash(img):
    """get image pHash value"""
    # 加载并调整图片为32x32灰度图片
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32, 32)
    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])


#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

