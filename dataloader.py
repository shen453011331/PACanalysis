# cv

import cv2
import time
import pandas as pd
import numpy as np


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

    def load_R(self, data_idx):
        start = time.clock()
        filename = '%s/%06d.jpg' % (self.data_path, data_idx)
        self.image = cv2.imread('%s/%06d.jpg'%(self.data_path, data_idx), cv2.IMREAD_GRAYSCALE)
        self.fImage = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        self.eqI = cv2.equalizeHist(self.image)
        elapsed = time.clock() - start
        return self.image, self.fImage, elapsed, self.eqI
        # return self.image, self.fImage, self.eqI, elapsed


    # def load_left_horn_points(self, csv_file):
    #     df = pd.read_csv(csv_file, sep=' ', index_col=False, header=None)
    #     df.columns = ['image_path', 'left/right', 'value', 'x1', 'y1', 'x2', 'y2']
    #     self.df_left = df

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
            pass
        else:
            return np.array([0, 0]), np.array([0, 0])