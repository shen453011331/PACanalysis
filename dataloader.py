# cv

import cv2
import time


class DataLoader(object):

    def __init__(self, data_path):
        self.data_path = data_path

    def load(self, data_idx):
        start = time.clock()
        self.image = cv2.imread('%s/%06d.bmp'%(self.data_path, data_idx), cv2.IMREAD_GRAYSCALE)
        self.fImage = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        self.eqI = cv2.equalizeHist(self.image)
        elapsed = time.clock() - start
        return self.image, self.fImage, elapsed, self.eqI
        # return self.image, self.fImage, self.eqI, elapsed
