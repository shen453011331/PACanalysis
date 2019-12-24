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






