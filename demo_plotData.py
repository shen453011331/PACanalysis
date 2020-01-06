# 2020.1.4 this is for plot the analysis value to determine what to do next

from dataloader import *
from contactlocate import *
from plotresult import *
from result_saving import *
from process3D import *
from analysis_data import *


system_path = 'D:/PACanalysis/'
result_dir = 'run_data/'
result_value = system_path + result_dir + 'result_value.csv'
analysis = system_path + result_dir + 'analysis.csv'

plot = PlotResult()

if __name__ == '__main__':
    number = 5000
    # plot dist1 in left
    df = pd.read_csv(analysis)
    x_data = df['number'].values
    x_data = x_data.tolist()
    y_data = df['dist1_l'].values
    y_data[y_data>50] = 0
    y_data = y_data.tolist()

    plot.plot_lines(x_data, y_data, 'dist1_l.html')

