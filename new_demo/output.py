# this file define the result for the tracking results and two funcion to save and load the result
# 2020.8.21
import pandas as pd
import pickle
import os


def save_result(result, filename):
    output_hal = open(filename, 'wb')
    str_chan = pickle.dumps(result)
    output_hal.write(str_chan)
    output_hal.close()


def load_result(filename):
    with open(os.path.join(os.path.curdir, 'result', filename), 'rb') as file:
        return pickle.loads(file.read())


class Result(object):
    def __init__(self):
        self.res = pd.DataFrame(columns=['x', 'y', 'w', 'h'])
        self.time = pd.DataFrame(columns=['t'])

    def update(self, roi, t):
        self.res = self.res.append({'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3]}, ignore_index=True)
        self.time = self.time.append({'t': t}, ignore_index=True)