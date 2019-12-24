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

    def __init__(self, locate_params, track_params):
        self.locate_params = locate_params
        self.track_params = track_params
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
        # result
        self.locatePoints = None
        self.contactL = None
        self.contactR = None
        self.refinePoints = None
        self.sumLocate = 0

        # track
        self.tracker = cv2.TrackerKCF_create()
        self.wintx = 9
        self.winty = 9
        self.oldImage = None
        self.trackLength = 0  # 控制跟踪序列init更新间距

        # verify
        self.pointsDisT = [10, 35]

        # bool
        self.b_track = False
        self.b_init = False
        self.bbox = None

        # show
        self.show = None

    def corner_locating_no_verify(self, image, fImage, index):
        start = time.clock()
        if not self.b_track:
            points = self.do_locate(fImage)
            points = self.do_refine(fImage, points)
            # if verify():
            #     pass
            self.b_track = True
            if index%100 == 0:
                print('Num %d do locating'%index)
        else:
            points = self.do_track(image, self.locatePoints)
            points = self.do_refine(fImage, points)
            if index % 100 == 0:
                print('Num %d do tracking' % index)
        elapsed = time.clock() - start
        return points, elapsed
        pass

    def corner_locating_with_verify(self, image, fImage, index):
        start = time.clock()
        if not self.b_track:
            points = self.do_locate(fImage)
            points_r = self.do_refine(fImage, points)
            self.b_track = True
            if index % 100 == 0:
                print('Num %d do locating' % index)
        else:
            points = self.do_track(image, self.locatePoints)
            points_r = self.do_refine(fImage, points)
            if index % 100 == 0:
                print('Num %d do tracking' % index)
        if not self.verify(points):
            self.b_track = False
            print("locate Error #%d" % index)
            points = np.zeros((2, 2), dtype='float')
            points_r = points
        self.locatePoints = points
        elapsed = time.clock() - start
        return points_r, elapsed
        pass

    def corner_locating_with_update(self, image, fImage, index):
        start = time.clock()
        self.thisislocate = False
        if not self.b_track:
            self.trackLength = 0
            self.thisislocate = True
            points = self.do_locate(fImage)
            points_r = self.do_refine(fImage, points)
            self.b_track = True
            self.b_init = False  # to init tracking
            if index % 100 == 0:
                print('Num %d do locating' % index)
        else:
            points = self.do_track(self.oldImage, image, self.locatePoints)
            points_r = self.do_refine(fImage, points)
            if index % 100 == 0:
                print('Num %d do tracking' % index)
            self.trackLength = self.trackLength + 1
            if self.trackLength > 500:
                self.trackLength = 0
                self.b_init = False
        if not self.verify(fImage, points, points_r):
            self.b_track = False

            print("locate Error #%d" % index)
            points = np.zeros((2, 2), dtype='float')
            points_r = points
        elif self.thisislocate:
            self.getPara(fImage)
        self.locatePoints = points
        self.oldImage = image
        elapsed = time.clock() - start
        return points_r, points, elapsed
        pass

    def do_multiTrack_update(self, image, fImage, index):
        start = time.clock()
        self.thisislocate = False

        if not self.b_track:
            self.thisislocate = True
            points = self.do_locate(fImage)
            self.points_r = self.do_refine(fImage, points)
            self.b_track = True
            self.b_init = False  # to init tracking
            if index % 100 == 0:
                print('Num %d do locating' % index)
        else:
            points = self.do_local_track(self.oldImage, image, self.points_r)
            self.points_r = self.do_refine(fImage, points)
            if index % 100 == 0:
                print('Num %d do tracking' % index)
        if not self.verify(fImage, points, self.points_r):
            self.b_track = False

            print("locate Error #%d" % index)
            points = np.zeros((2, 2), dtype='float')
            self.points_r = points
        elif self.thisislocate:
            self.getPara(fImage)
        self.locatePoints = points
        self.oldImage = image
        elapsed = time.clock() - start
        return self.points_r, points, elapsed
        pass

    def do_locate(self, fImage):
        LineROI = [0, 400, 300, 2100]
        PanROI = [300, 550, 800, 1600]  #  300 550
        self.updateTemplate()
        self.locateImage(LineROI, PanROI, fImage)
        self.doCrossingPoints(self.lftpoint, self.rgtpoint, self.ptgpoint)
        return self.locatePoints
        pass

    def do_track(self, oldImage, image, points):
        self.getRectFromPoints(points)
        if not self.b_init:
            self.tracker = None
            self.tracker = cv2.TrackerKCF_create()
            ok = self.tracker.init(oldImage, self.bbox)
            ok, self.bbox = self.tracker.update(oldImage)
            self.b_init = True

        ok, self.bbox = self.tracker.update(image)
        self.getPointsFromRect()
        pass
        return self.trackPoints

    def do_local_track(self, oldImage, image, points_l):
        self.getRectFromPoints(points_l)
        local_bbox_l = self.getLocalRect(points_l[0, :], self.bbox)
        local_bbox_r = self.getLocalRect(points_l[1, :], self.bbox)
        old_patch = self.get_patch(oldImage, self.bbox)
        if not self.b_init:
            self.tracker = None
            self.tracker = cv2.TrackerKCF_create()
            ok = self.tracker.init(oldImage, self.bbox)
            ok, self.bbox = self.tracker.update(oldImage)
            self.localtracker_l = None
            self.localtracker_l = cv2.TrackerKCF_create()
            ok = self.localtracker_l.init(old_patch, local_bbox_l)
            ok, self.bbox_local_l = self.localtracker_l.update(old_patch)
            self.localtracker_r = None
            self.localtracker_r = cv2.TrackerKCF_create()
            ok = self.localtracker_r.init(old_patch, local_bbox_r)
            ok, self.bbox_local_r = self.localtracker_r.update(old_patch)
            self.b_init = True
        ok, self.bbox = self.tracker.update(image)
        new_patch = self.get_patch(image, self.bbox)
        ok_l, self.bbox_local_l = self.localtracker_l.update(new_patch)
        ok_r, self.bbox_local_r = self.localtracker_r.update(new_patch)
        # self.show.plot_test_patch(new_patch, self.bbox_local_l, self.bbox_local_r)
        self.trackPoints = points_l
        if ok_l and ok_r:
            self.trackPoints[0, 0] = self.bbox_local_l[0] + self.bbox_local_l[2] / 2 + self.bbox[0]
            self.trackPoints[0, 1] = self.bbox_local_l[1] + self.bbox_local_l[3] / 2 + self.bbox[1]
            self.trackPoints[1, 0] = self.bbox_local_r[0] + self.bbox_local_r[2] / 2 + self.bbox[0]
            self.trackPoints[1, 1] = self.bbox_local_r[1] + self.bbox_local_r[3] / 2 + self.bbox[1]
        else:
            bbox = self.bbox
            midPoints = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            midPoints = np.array(midPoints, dtype=float)
            shift = np.array(self.shift, dtype=float)
            if not ok_l:
                self.trackPoints[0, :] = midPoints - shift
            if not ok_r:
                self.trackPoints[1, :] = midPoints + shift
        # self.getPointsFromRect()
        pass
        return self.trackPoints

    def get_patch(self, image, bbox):
        x1 = int(bbox[0])
        x2 = int(bbox[0] + bbox[2])
        y1 = int(bbox[1])
        y2 = int(bbox[1] + bbox[3])
        return image[y1:y2, x1:x2]

    def do_refine(self, fImage, points):
        floatI = np.float32(fImage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(floatI, np.float32(points), (self.wintx, self.winty), (-1, -1), criteria)
        self.refinePoints = corners
        return self.refinePoints

    def do_params_update(self):
        pass

    def locateImage(self, roi_line, roi_pan, fImage):
        # 根据的接触线和受电弓区域进行模板匹配，这里区域都为矩形，且一旦相机位置不变，该区域不变

        I_Line = fImage[roi_line[0]:roi_line[1], roi_line[2]:roi_line[3]]
        I_Pan = fImage[roi_pan[0]:roi_pan[1], roi_pan[2]:roi_pan[3]]
        step = round(I_Line.shape[0] / (self.numpoint - 1))
        method = cv2.TM_CCORR_NORMED

        # do line template matching for contact wire
        for i in range(self.numpoint):
            yi = int(roi_line[0] + i * step)

            Li = fImage[int(yi):int(yi + 1), int(roi_line[2]):int(roi_line[3])]
            Li = Li - self.meangrayr * np.ones([1, I_Line.shape[1]])
            Li = Li.astype(np.float32)
            res = cv2.matchTemplate(Li, self.imapatchr, method)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            di = max_loc[0]
            self.lftpoint[:, i] = [int(di + self.widpatchr - 1 + int(roi_line[2]) - 1), yi]
            self.rgtpoint[:, i] = [int(di + 2 * self.widpatchr - 2 + int(roi_line[2]) - 1), yi]

        step = round(I_Pan.shape[1] / (self.numpoint - 1))
        for i in range(self.numpoint):
            xi = int(roi_pan[2] + i * step)
            Li = fImage[roi_pan[0]:roi_pan[1], xi:xi + 1].T
            Li = Li - self.meangrayc * np.ones([1, I_Pan.shape[0]])
            Li = Li.astype(np.float32)
            res = cv2.matchTemplate(Li, self.imapatchc, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            di = max_loc[0]
            self.ptgpoint[:, i] = [xi, int(di + roi_pan[0] + self.widpatchc - 2)]  # -4是为了保证可以更容易优化到角点

    def updateTemplate(self):
        # 为后续模板参数更新做准备，将模板根据参数重新调整
        self.imapatchr = np.zeros([1, self.widpatchr * 3], dtype='float32')
        self.imapatchr[0, self.widpatchr:2 * self.widpatchr] -= np.ones(self.widpatchr) * self.disr / 255
        self.imapatchc = np.zeros([1, self.widpatchc * 2], dtype='float32')
        self.imapatchc[0, self.widpatchc:2 * self.widpatchc] -= np.ones(self.widpatchc) * self.disc / 255

    def doCrossingPoints(self, lftpoint, rgtpoint, ptgpoint):
        # 根据模板匹配获取的点集进行直线交错
        self.kl, bl, self.il = ransac(lftpoint, 5)
        self.kr, br, self.ir = ransac(rgtpoint, 5)
        self.kh, bh, self.ih = ransac(ptgpoint, 10)
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
        self.locatePoints = np.vstack((xl, xr))
        self.contactL = xl
        self.contactR = xr

    def getRectFromPoints(self, points):
        widthx = 100
        heighty = 100
        left = int(np.mean(points[:, 0]) - widthx / 2)
        up = int(np.mean(points[:, 1]) - heighty / 2)
        shiftx = points[1, 0] - np.mean(points[:, 0])
        shifty = points[1, 1] - np.mean(points[:, 1])
        self.shift = (shiftx, shifty)
        self.bbox = (left, up, widthx, heighty)
        self.trackPoints = points

    def getLocalRect(self, points, bbox):
        widthx = 30
        heighty = 30
        left = int(points[0] - widthx / 2) - bbox[0]
        up = int(points[1] - heighty / 2) - bbox[1]
        # shiftx = points[1, 0] - np.mean(points[:, 0])
        # shifty = points[1, 1] - np.mean(points[:, 1])
        # self.shift = (shiftx, shifty)
        return (left, up, widthx, heighty)

    def getPointsFromRect(self):
        bbox = self.bbox
        midPoints = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))

        midPoints = np.array(midPoints, dtype=float)
        shift = np.array(self.shift, dtype=float)
        self.trackPoints[0, :] = midPoints - shift
        self.trackPoints[1, :] = midPoints + shift

    def verify(self, fImage, points, points_r):
        # err3 左右接触点位置判断
        if points[0, 0] <= 0 or points[0, 1] <= 0 \
                or points[1, 0] <= 0 or points[1, 1] <= 0\
                or points[0, 0] > imageWidth or points[1, 0] > imageWidth \
                or points[0, 1] > imageHeight or points[1, 1] > imageHeight:
            print('points:')
            print(points)
            return False
        err3 = abs(points[0, 0] - points[1, 0])
        if not self.pointsDisT[0] <= err3 <= self.pointsDisT[1]:
            print('err3:%f' % err3)
            print(points)
            return False
        # # err4 判断角点是否为角点，简单判断，窗口右上角和左右中心点的灰度差
        # x_mid = int(points_r[0, 0])
        # y_mid = int(points_r[0, 1])
        # x_le_up = int(points_r[0, 0] -self.wintx)
        # y_le_up = int(points_r[0, 1] - self.winty)
        # x_rg_up = int(points_r[1, 0] + self.wintx)
        # y_rg_up = int(points_r[1, 1] - self.winty)
        # rgb1 = fImage[y_mid, x_mid]
        # rgb2 = fImage[y_le_up, x_le_up]
        # rgb3 = fImage[y_rg_up, x_rg_up]
        # dis1 = abs(rgb1 - rgb2)
        # dis2 = abs(rgb1 - rgb3)
        # if dis1 < 0.1 or dis2 < 0.1:
        #     return False
        return True

    def verify_hash(self, fImage, points, points_r):
        # err3 左右接触点位置判断
        if points[0, 0] <= 0 or points[0, 1] <= 0 \
                or points[1, 0] <= 0 or points[1, 1] <= 0\
                or points[0, 0] > imageWidth or points[1, 0] > imageWidth \
                or points[0, 1] > imageHeight or points[1, 1] > imageHeight:
            print('points:')
            print(points)
            return False
        err3 = abs(points[0, 0] - points[1, 0])
        if not self.pointsDisT[0] <= err3 <= self.pointsDisT[1]:
            print('err3:%f' % err3)
            print(points)
            return False


        # # err4 判断角点是否为角点，简单判断，窗口右上角和左右中心点的灰度差
        # x_mid = int(points_r[0, 0])
        # y_mid = int(points_r[0, 1])
        # x_le_up = int(points_r[0, 0] -self.wintx)
        # y_le_up = int(points_r[0, 1] - self.winty)
        # x_rg_up = int(points_r[1, 0] + self.wintx)
        # y_rg_up = int(points_r[1, 1] - self.winty)
        # rgb1 = fImage[y_mid, x_mid]
        # rgb2 = fImage[y_le_up, x_le_up]
        # rgb3 = fImage[y_rg_up, x_rg_up]
        # dis1 = abs(rgb1 - rgb2)
        # dis2 = abs(rgb1 - rgb3)
        # if dis1 < 0.1 or dis2 < 0.1:
        #     return False
        return True

    def getPara(self, fImage):
        # 根据匹配的模板获取参数
        sum_max = 0
        sum_min = 0
        sum_num = 0
        for i in range(len(self.il)):
            if (self.il[i] == True):

                temp_y = self.lftpoint[1, i]
                #         print(temp_y)
                if (temp_y < self.locatePoints[0, 1]):
                    temp_x = self.lftpoint[0, i] - self.widpatchr
                    #         print(temp_y,temp_x)
                    temp = fImage[temp_y:temp_y + 1, temp_x:temp_x + self.widpatchr * 3]
                    #         print(temp.shape)
                    list_temp = temp.T.tolist()
                    max_temp = max(list_temp)
                    min_temp = min(list_temp)
                    sum_max += max_temp[0]
                    sum_min += min_temp[0]
                    sum_num += 1
        #             print(max_temp,min_temp)
        mean_max = sum_max / sum_num
        mean_min = sum_min / sum_num
        self.meangrayr = mean_max
        self.disr = round((mean_max - mean_min) * 256)
        # sum_max = 0
        # sum_min = 0
        # sum_num = 0
        # for i in range(len(self.ih)):
        #     if (self.ih[i] == True):
        #         temp_y = self.ptgpoint[1, i] - self.widpatchc
        #         temp_x = self.ptgpoint[0, i]
        #         if (self.locatePoints[0, 0] < temp_x < self.locatePoints[1, 0]):
        #             temp_noneuse_a = 0
        #         else:
        #             temp = fImage[temp_y:temp_y + self.widpatchc * 2, temp_x:temp_x + 1]
        #             list_temp = temp.tolist()
        #             max_temp = max(list_temp)
        #             min_temp = min(list_temp)
        #             sum_max += max_temp[0]
        #             sum_min += min_temp[0]
        #             sum_num += 1
        # #             print(max_temp,min_temp)
        # mean_max = sum_max / sum_num
        # mean_min = sum_min / sum_num
        # self.meangrayc = mean_max
        # self.disc = round((mean_max - mean_min) * 256)
        self.meangrayc = self.meangrayr
        self.disc = self.disr
        print ('parameter changed')
        print('mean: %f, dis: %f'%(self.meangrayr, self.disr))


def ransac(points, threshold):
    # 进行ransac，用于保障定位的鲁棒性
    ransac_model = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=3,
                                   loss='squared_loss', stop_n_inliers=8,
                                   residual_threshold=threshold, random_state=0)
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

