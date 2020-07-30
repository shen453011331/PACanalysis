# this file is used to track contact points
import cv2
import time
import numpy as np

class TrackParams(object):
    def __init__(self, widthx, heighty):
        self.heighty = heighty
        self.widthx = widthx
        self.patch_l_tgt, self.patch_r_tgt = None, None
        self.hash_thre = 50

    def update_patch(self, patch_l_tgt, patch_r_tgt, hash_thre=50):
        self.patch_l_tgt, self.patch_r_tgt = patch_l_tgt, patch_r_tgt
        self.hash_thre = hash_thre


class OutputTrack(object):
    def __init__(self, params):
        self.params = params
        self.result_list = []
        self.index_list = []

    def save_result(self, index, rect, shift, points_l, points_r):
        self.index_list.append(index)
        result = dict({'index': index, 'rect': rect, 'shift': shift,
                       'points_l': points_l, 'points_r': points_r})
        self.result_list.append(result)


class TrackerPAC(object):
    def __init__(self, params):
        self.params = params
        # track
        self.tracker = cv2.TrackerKCF_create()
        self.track_length = 0  # 控制跟踪序列init更新间距
        self.localtracker_l = cv2.TrackerKCF_create()
        self.localtracker_r = cv2.TrackerKCF_create()
        self.output_track = OutputTrack(params)
        self.b_init = False
        self.bbox = None
        self.shift = None
        self.hash = []

    def do_track_refine(self, index, oldImage, image, points=None, resize=0, b_re_init=False):
        start_time = time.clock()
        if b_re_init:
            self.b_init = False
        if points is not None:
            self.getRectFromPoints(points)
        old_bbox = self.bbox
        if not self.b_init:
            self.tracker = None
            self.tracker = cv2.TrackerKCF_create()
            ok = self.tracker.init(oldImage, self.bbox)
            ok, self.bbox = self.tracker.update(oldImage)
            self.b_init = True
        ok, self.bbox = self.tracker.update(image)
        points = self.getPointsFromRect()
        points_temp = do_refine(image, points)
        points = choose_refine(points, points_temp)
        self.output_track.save_result(index, self.bbox, self.shift, points[0, :], points[1, :])
        elapsed = time.clock() - start_time
        return points, ok, elapsed

    def do_track_refine_hash(self, index, oldImage, image, points=None, resize=0, b_re_init=False):
        start_time = time.clock()
        if b_re_init:
            self.b_init = False
        if points is not None:
            self.getRectFromPoints(points)
        old_bbox = self.bbox
        if not self.b_init:
            self.tracker = None
            self.tracker = cv2.TrackerKCF_create()
            ok = self.tracker.init(oldImage, self.bbox)
            ok, self.bbox = self.tracker.update(oldImage)
            self.b_init = True
        ok, self.bbox = self.tracker.update(image)
        points = self.getPointsFromRect()
        # once the tracking position is get ,we need to do hash value for verify
        patch_l_tgt, patch_r_tgt = self.params.patch_l_tgt, self.params.patch_r_tgt
        patch_l, patch_r = get_patch(image, points[0:1, :]), get_patch(image, points[1:, :])
        hash_l = cmpHash(pHash(patch_l), pHash(patch_l_tgt))
        hash_r = cmpHash(pHash(patch_r), pHash(patch_r_tgt))
        self.hash = [hash_l, hash_r]
        # if 0:
        if max(hash_l, hash_r) > self.params.hash_thre:
            # print('hash verify error l:{}, r:{}'.format(hash_l, hash_r))
            ok = False
            # 额外进行一次判定是否存在电弧, 通过hash和patch亮度联合判断
            if (len(patch_l[patch_l > 200]) > 50 or len(patch_r[patch_r > 200]) > 50) and max(hash_l, hash_r) > 100:
                print('NO.{} has spark, continue tracking'.format(index))
                ok = True
                points = self.get_simulate_points(index)

        if ok:
            points_temp = do_refine(image, points)
            points = choose_refine(points, points_temp)
        self.output_track.save_result(index, self.bbox, self.shift, points[0, :], points[1, :])
        elapsed = time.clock() - start_time
        return points, ok, elapsed

    def getRectFromPoints(self, points):
        widthx = self.params.widthx
        heighty = self.params.heighty
        left = int(np.mean(points[:, 0]) - widthx / 2)
        up = int(np.mean(points[:, 1]) - heighty / 2)
        shiftx = points[1, 0] - np.mean(points[:, 0])
        shifty = points[1, 1] - np.mean(points[:, 1])
        self.shift = (shiftx, shifty)
        self.bbox = (left, up, widthx, heighty)

    def getPointsFromRect(self):
        bbox = self.bbox
        mid_points = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        mid_points = np.array(mid_points, dtype=float)
        shift = np.array(self.shift, dtype=float)
        track_points = np.zeros([2, 2], dtype='float')
        track_points[0, :] = mid_points - shift
        track_points[1, :] = mid_points + shift
        return track_points

    def get_simulate_points(self, idx):
        last_index, last_index_2 = idx - 1, idx - 2
        points_last = get_points_from_output(self.output_track, last_index)
        points_last_2 = get_points_from_output(self.output_track, last_index_2)
        if points_last[0, 0] == 0 or points_last_2[0, 0] == 0:
            points = points_last
        else:
            points = points_last * 2 - points_last_2
        return points

def do_refine(img, points):
    wintx, winty = 9, 9
    points_temp = np.float32(points).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    # img and points must be float32
    corners = cv2.cornerSubPix(img, points_temp, (wintx, winty), (-1, -1), criteria)
    return corners


def choose_refine(points, points_r):
    dis = points - points_r
    # print('l-move {:.2f},{:.2f}'.format(dis[0, 0], dis[0, 1]))
    # print('r-move {:.2f},{:.2f}'.format(dis[1, 0], dis[1, 1]))
    dis_r = np.linalg.norm(points_r[0, :]-points_r[1, :])
    # print('dis between refine is {}'.format(dis_r))
    b_change = 1 if np.max(np.abs(dis)) < 3 and 30 > dis_r > 12 else 0
    return points_r if b_change else points


def pHash(img):
    """get image pHash value"""
    # 加载并调整图片为32x32灰度图片
    # img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    # 创建二维列表
    vis0 = img.astype(np.float32)
    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    # vis1.resize(32, 32)
    # 把二维list变成一维list
    img_list = vis1.flatten()
    # 计算均值
    avg = np.mean(img_list)
    avg_list = img_list
    avg_list[img_list < avg] = 0
    avg_list[img_list >= avg] = 1
    new_array = avg_list.reshape(-1, 4)
    new_array2 = new_array[:, 3] + 2 * new_array[:, 2] + 4 * new_array[:, 1] + 8 * new_array[:, 0]
    new_array3 = new_array2.tolist()
    new_array3 = ['%x' % int(i) for i in new_array3]
    hash_value = ''.join(new_array3)
    # 得到哈希值
    return hash_value


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        n = n+1 if hash1[i] != hash2[i] else n
    return n


def get_patch(image, point, size=(32, 32)):
    x, y = int(point[0, 0]), int(point[0, 1])
    w, h = int(size[0]/2), int(size[1]/2)
    patch = image[y-h: y+h, x-w: x+w]
    patch = image[:h*2, :w*2] if patch.shape != (32, 32) else patch
    return patch


def get_points_from_output(output, index):
    if index in output.index_list:
        index_last = output.index_list.index(index)
        dict_data = output.result_list[index_last]
        points = np.vstack((dict_data['points_l'], dict_data['points_r']))
    else:
        points = np.zeros((2, 2), dtype=float)
    return points