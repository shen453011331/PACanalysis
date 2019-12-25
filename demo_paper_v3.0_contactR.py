# 试图用一个纯跟踪问题去解决，通过多级跟踪，先忽略速度的情况，实现稳定接触点定位
import argparse
import cv2
import time
from dataloader import *
from contactlocate import *
from plotresult import *
from result_saving import *

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_path',  dest='data_path', type=str, default='F:/dataset/R/JPEGImages', help='path of data')
parser.add_argument('--data_num',  dest='data_num', type=int, default=21732, help='number of data')
parser.add_argument('--meangrayr',  dest='meangrayr', type=float, default=80.0/255, help='line template 1')
parser.add_argument('--meangrayc',  dest='meangrayc', type=float, default=88.0/255, help='horn template 1')
parser.add_argument('--widpatchr',  dest='widpatchr', type=int, default=18, help='line template 2')
parser.add_argument('--widpatchc',  dest='widpatchc', type=int, default=10, help='horn template 2')
parser.add_argument('--disr',  dest='disr', type=int, default=55, help='line template 3')
parser.add_argument('--disc',  dest='disc', type=int, default=64, help='horn template 3')
parser.add_argument('--thre',  dest='thre', type=int, default=25, help='max threshold for verify')

args = parser.parse_args()
ldata = DataLoader(args.data_path)
locate_params = {'meangrayr': args.meangrayr,
                 'meangrayc': args.meangrayc,
                 'widpatchr': args.widpatchr,
                 'widpatchc': args.widpatchc,
                 'disr': args.disr,
                 'disc': args.disc}

track_params = {}
refine_params = {}
verify_params = {'thre': args.thre}
rprocess = ContactLocate(locate_params, track_params, refine_params, verify_params)

testShow = PlotResult()
saveResult = SaveResult('C:/Users/Administrator/Documents/sy_paper_contactAnalysis/new/')
if __name__ == '__main__':
    sumTime = 0
    rprocess.show = testShow
    # 1 .仅仅locate
    for i in range(4750, args.data_num):
        image, fImage, _ , _= ldata.load_R(i + 1)
        start = time.clock()
        points = rprocess.do_locate(fImage)
        # points_r = lprocess.do_refine(fImage, points)
        elapsed = time.clock() - start
        sumTime = sumTime + elapsed
        fps = (i + 1) / sumTime
        testShow.plot_locate_result(image, points, 1)
        saveResult.saveResult(points, (i + 1), fps, 'locate+refine_R.csv')

    # # 2. locate+ 第一步track
    #
    # for i in range(1000):
    #     image, fImage, _ = ldata.load(i + 1)
    #     start = time.clock()
    #     if not i:# 这里之后可以附加上新的置信度条件
    #         points = lprocess.do_locate(fImage)
    #         points_r = lprocess.do_refine(fImage, points)
    #     else:
    #         points = lprocess.do_track(lprocess.oldImage, image, points)
    #         points_r = lprocess.do_refine(fImage, points)
    #     image = lprocess.oldImage
    #     elapsed = time.clock() - start
    #     sumTime = sumTime + elapsed
    #     fps = (i + 1) / sumTime
    #     if i%100 == 0:
    #         print('%d image has been processed' % (i+1))
    #     saveResult.saveResult(points_r, (i + 1), fps, '2.locate+refine+1.track.csv')

    # # 3. 增加局部的track
    # for i in range(4750, args.data_num):
    #     # lprocess.meangrayr = 50
    #     # lprocess.disr = 3
    #     image, fImage, _, eqI = ldata.load_R(i + 1)
    #
    #     points, points_l, time_process = rprocess.do_multiTrack_update(image, fImage, i + 1)
    #     testShow.plot_locate_result(image, points, 1)
    #     sumTime = sumTime + time_process
    #     fps = (i + 1) / sumTime
    #     if i % 100 == 0:
    #         print('%d image has been processed' % (i + 1))
    #     saveResult.saveResult(points, (i + 1), fps, 'temp2800.csv')






    # 无用
    # # 根据给定点跟踪
    # _, fImage, _, lprocess.oldImage = ldata.load(7595)
    # p_7595 = np.array([[1144.05, 388.58], [1123.69, 391.33]], dtype='float')
    # image, fImage, _, eqI = ldata.load(7596)
    # lprocess.b_track = True
    # lprocess.b_init = False
    # lprocess.points_r = lprocess.do_local_track(lprocess.oldImage, eqI, p_7595)
    # points = lprocess.do_refine(fImage, lprocess.points_r)
    # lprocess.oldImage = eqI
    # for i in range(7596, 7700):
    #     # lprocess.meangrayr = 50
    #     # lprocess.disr = 3
    #     image, fImage, _, eqI = ldata.load(i + 1)
    #     lprocess.points_r = lprocess.do_local_track(lprocess.oldImage, eqI, lprocess.points_r)
    #     points = lprocess.do_refine(fImage, lprocess.points_r)
    #     lprocess.oldImage = eqI
    #     testShow.plot_locate_result(image, points, 0)
    #     sumTime = sumTime + 0.02
    #     fps = (i + 1) / sumTime
    #     if i % 100 == 0:
    #         print('%d image has been processed' % (i + 1))
    #     saveResult.saveResult(points, (i + 1), fps, 'temp2800.csv')

