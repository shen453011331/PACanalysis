# this file can load locate output files and output image with lines and circles with fix index
import argparse
import cv2
import time
import os
from dataloader import *
from contactlocate import *
from plotresult import *
from result_saving import *
from process3D import *
from analysis_data import *
from locate import *
import pickle
parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_pathL',  dest='data_pathL', type=str, default='F:/dataset/L', help='path of data')
args = parser.parse_args()

ldata = DataLoader(args.data_pathL)

if __name__ == '__main__':
    idx_list = [1, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 20, 36, 37, 39, 405, 409, 411, 415, 426, 570, 577, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 644, 646, 647, 648, 649, 650, 651, 654, 655, 656, 659, 662, 663, 664, 666, 669, 670, 674, 675, 676, 677, 681, 694, 712, 714, 722, 748, 749, 757, 765, 769, 771, 773, 774, 775, 793, 806, 819, 820, 937, 967, 976, 986, 992, 999, 1004, 1009, 1027, 1193, 1195, 1233, 1234, 1237, 1251, 1281, 1340, 1344, 1347, 1367, 1368, 1369, 1371, 1376, 1396, 1422, 1433, 1434, 1439, 1440, 1441, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1464, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1572, 1577, 1582, 1600, 1602, 1605, 1621, 1645, 1790, 1808, 1819, 1825, 1826, 1924, 1955, 2015, 2074, 2095, 2235, 2239, 2240, 2245, 2258, 2265, 2276, 2277, 2281, 2285, 2290, 2314, 2465, 2635, 2637, 2658, 2673, 2680, 2779, 2804, 2830, 2831, 2834, 2836, 2838, 2839, 2848, 2883, 2890, 2893, 2901, 2918, 2937, 2957, 2993, 3025, 3048, 3049, 3058, 3059, 3070, 3072, 3075, 3081, 3085, 3089, 3090, 3094, 3095, 3100, 3102, 3106, 3116, 3145, 3156, 3276, 3283, 3287, 3288, 3358, 3388, 3431, 3435, 3441, 3447, 3449, 3451, 3452, 3455, 3472, 3477, 3478, 3479, 3481, 3485, 3487, 3488, 3491, 3493, 3497, 3498, 3500, 3506, 3507, 3518, 3519, 3520, 3521, 3522, 3523, 3555, 3561, 3562, 3709, 3710, 3714, 3861, 3870, 3871, 3877, 3885, 3890, 3893, 3901, 4186, 4205, 4208, 4346, 4347, 4348, 4373, 4506, 4512, 4514, 4515, 4517, 4518, 4519, 4520, 4528, 4537, 4542, 4691, 4697, 4725, 4727, 4755, 4763, 4896, 4906]

    # idx_list = [i+1 for i in range(20)]
    system_path = 'D:/clean/PACanalysis/'

    testShow = PlotResult()
    gt_file_l = 'D:\PACanalysis\gt'
    df_gt = ldata.load_gt_l(gt_file_l, 5000)
    title = 'locate_refine_update'
    with open(os.path.join(os.path.join(system_path, 'result', title, 'test_locate.pkl')), 'rb') as file:
        locate_output = pickle.loads(file.read())
    output_path = os.path.join(system_path, 'output', title)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for idx in idx_list:
        output_name = os.path.join(output_path, '{:06d}.png'.format(idx))
        filepath = os.path.join('F:/dataset/L', '{:06d}.bmp'.format(idx))
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = testShow.draw_locate(img, locate_output, idx)
        cv2.imwrite(output_name, img)