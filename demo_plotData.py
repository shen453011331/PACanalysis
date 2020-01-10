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
    plot.plot_line_path = 'D:/PACanalysis/run_data/html/'
    # plot dist1 in left
    df = pd.read_csv(analysis)
    x_data = df['number'].values
    x_data = x_data.tolist()
    y_data = df['dist1_l'].values
    y_data[y_data>50] = 0
    y_data = y_data.tolist()
    #  plot dist1
    plot.plot_lines(x_data, y_data, 'dist1_l.html')
    y_data = df['dist1_r'].values
    y_data[y_data > 50] = 0
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist1_r.html')
    # plot dist2
    y_data = df['dist2_l_l'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist2_l_l.html')
    y_data = df['dist2_l_r'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist2_l_r.html')
    y_data = df['dist2_l_l'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist2_r_l.html')
    y_data = df['dist2_l_r'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist2_r_r.html')
    # plot dist 3
    y_data = df['dist3_l'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist3_l.html')
    y_data = df['dist3_r'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist3_r.html')
    # plot speed error
    y_data = df['speed_err_l'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'speed_err_l.html')
    y_data = df['speed_err_r'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'speed_err_r.html')
    # plot shift1 and dist4
    y_data = df['shift1'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'shift1.html')
    y_data = df['dist4'].values
    y_data = y_data.tolist()
    plot.plot_lines(x_data, y_data, 'dist4.html')

