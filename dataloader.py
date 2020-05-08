# cv

import cv2
import time
import pandas as pd
import numpy as np


class DataLoader(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.image, self.fImage, self.eqI = None, None, None
        self.df = None  # df of horn locate results
        self.l_gt = None  # df of locating ground truth

    def load(self, data_idx):
        start = time.clock()
        self.image = cv2.imread('%s/%06d.bmp' % (self.data_path, data_idx), cv2.IMREAD_GRAYSCALE)
        self.fImage = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        self.fImage = self.fImage.astype(np.float32)
        self.eqI = cv2.equalizeHist(self.image)
        elapsed = time.clock() - start
        return self.image, self.fImage, elapsed, self.eqI
        # return self.image, self.fImage, self.eqI, elapsed

    def load_R(self, data_idx):
        start = time.clock()
        self.image = cv2.imread('%s/%06d.jpg' % (self.data_path, data_idx), cv2.IMREAD_GRAYSCALE)
        self.fImage = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        self.fImage = self.fImage.astype(np.float32)
        self.eqI = cv2.equalizeHist(self.image)
        elapsed = time.clock() - start
        return self.image, self.fImage, elapsed, self.eqI

    def load_horn_files(self, csv_file):
        df = pd.read_csv(csv_file, sep=' ', index_col=False, header=None)
        df.columns = ['image_path', 'left/right', 'value', 'x1', 'y1', 'x2', 'y2']
        self.df = df

    def load_hornPoints_from_df(self, df, index):
        path = '/VOC2007/JPEGImages/%06d.jpg' % index
        temp_df = df[df['image_path'].isin([path])]
        if len(temp_df) == 2:
            temp2= temp_df[temp_df['left/right'] == 1]
            leftPoints = np.hstack([temp2.x1.values, temp2.y2.values])
            temp2 = temp_df[temp_df['left/right'] == 2]
            rightPoints = np.hstack([temp2.x2.values, temp2.y2.values])
            return leftPoints, rightPoints
        else:
            return np.array([0, 0]), np.array([0, 0])

    def load_gt_l(self, filepath, num_gt):
        csv_file = '%s/%06d.csv' %(filepath, 1*1000)
        df = pd.read_csv(csv_file)
        full_df = pd.DataFrame(columns=df.columns)
        csv_num = round(num_gt/1000)
        for i in range(csv_num):
            csv_file = '%s/%06d.csv' %(filepath, (i+1)*1000)
            df = pd.read_csv(csv_file)
            full_df = pd.concat([full_df, df])
        self.l_gt = full_df
        return full_df
