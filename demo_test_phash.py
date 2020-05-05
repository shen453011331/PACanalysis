import argparse
import cv2
import time
import os
from dataloader import *
from plotresult import *
from result_saving import *
from process3D import *
from analysis_data import *
from locate import *
from tracker import *
import pickle

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_pathL',  dest='data_pathL', type=str, default='F:/dataset/L', help='path of data')
parser.add_argument('--data_pathR',  dest='data_pathR', type=str, default='F:/dataset/R/JPEGImages', help='path of data')

parser.add_argument('--data_num',  dest='data_num', type=int, default=21732, help='number of data')
parser.add_argument('--meangrayr',  dest='meangrayr', type=float, default=90.0/255, help='line template 1')
parser.add_argument('--meangrayc',  dest='meangrayc', type=float, default=88.0/255, help='horn template 1')
parser.add_argument('--widpatchr',  dest='widpatchr', type=int, default=24, help='line template 2')
parser.add_argument('--widpatchc',  dest='widpatchc', type=int, default=10, help='horn template 2')
parser.add_argument('--disr',  dest='disr', type=int, default=70, help='line template 3')
parser.add_argument('--disc',  dest='disc', type=int, default=64, help='horn template 3')
parser.add_argument('--thre',  dest='thre', type=int, default=25, help='max threshold for verify')
args = parser.parse_args()
ldata = DataLoader(args.data_pathL)
locate_params = {'meangrayr': args.meangrayr,
                 'meangrayc': args.meangrayc,
                 'widpatchr': args.widpatchr,
                 'widpatchc': args.widpatchc,
                 'disr': args.disr,
                 'disc': args.disc}
track_params = {}
refine_params = {}
verify_params = {'thre': args.thre}





if __name__ == '__main__':
    testShow = PlotResult()
    system_path = 'D:/clean/PACanalysis/'
    saveResult = SaveResult(system_path)
    pro3d = Process3D()
    ana_data = AnalysisData(saveResult, pro3d, testShow)

    gt_file_l = 'D:\PACanalysis\gt'
    num = 5000
    df_gt = ldata.load_gt_l(gt_file_l, num)

    title = 'track_refine'
    with open(os.path.join(system_path, 'result', title, 'test_track.pkl'), 'rb') as file:
        track_output = pickle.loads(file.read())
    with open(os.path.join(system_path, 'result', title, 'test_locate.pkl'), 'rb') as file:
        locate_output = pickle.loads(file.read())
    df = pd.DataFrame(columns={'index', 'hash_l', 'hash_r', 'fps', 'hash_l_10', 'hash_r_10'})
    patch_l_list, patch_r_list = [], []
    speed_list = []
    output_path = os.path.join(system_path, 'output', title, 'hash')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # df = pd.read_csv('hash_analysis.csv')
    patch_l_tgt, patch_r_tgt = None, None
    for i in range(827, num):
        img_idx = i + 1
        if img_idx == 1:
            filepath = os.path.join('F:/dataset/L', '{:06d}.bmp'.format(img_idx))
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            point_gt = df_gt[df_gt['number'] == img_idx]
            p_l = np.array(point_gt[['point_l_x', 'point_l_y']].values)
            p_r = np.array(point_gt[['point_2_x', 'point_2_y']].values)
            patch_l, patch_r = get_patch(img, p_l), get_patch(img, p_r)
            patch_l_tgt, patch_r_tgt = patch_l, patch_r
        # temp_df = df.iloc[i:i+1]
        # filepath = os.path.join('F:/dataset/L', '{:06d}.bmp'.format(img_idx))
        # img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # point_gt = df_gt[df_gt['number'] == img_idx]
        # p_l = np.array(point_gt[['point_l_x', 'point_l_y']].values)
        # p_r = np.array(point_gt[['point_2_x', 'point_2_y']].values)
        # patch_l, patch_r = get_patch(img, p_l), get_patch(img, p_r)
        # if temp_df['hash_r'].values[0] > 50 or temp_df['hash_l'].values[0] > 50:
        #     save_name = os.path.join(output_path, '{:06d}_hash_50.png'.format(img_idx))
        #     img_stack = np.vstack((np.hstack((patch_l,patch_r)), np.hstack((patch_l_tgt,patch_r_tgt))))
        #     cv2.imwrite(save_name, img_stack)


        if img_idx % 100 == 1:
            print('image No. {}'.format(img_idx))
        filepath = os.path.join('F:/dataset/L', '{:06d}.bmp'.format(img_idx))
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img_idx in track_output.index_list:
            # print(img_idx)
            index = track_output.index_list.index(img_idx)
            dict_data = track_output.result_list[index]
            points = np.vstack((dict_data['points_l'], dict_data['points_r']))
            # points[0, 0] = 20 if not 0 < points[0, 0]< img.shape[1] else points[0, 0]
            # points[0, 1] = 20 if not 0 < points[0, 1]< img.shape[1] else points[0, 1]
            # points[1, 0] = 20 if not 0 < points[1, 0]< img.shape[0] else points[1, 0]
            # points[1, 1] = 20 if not 0 < points[1, 1]< img.shape[0] else points[1, 1]

        # point_gt = df_gt[df_gt['number'] == img_idx]
        # p_l = np.array(point_gt[['point_l_x', 'point_l_y']].values)
        # p_r = np.array(point_gt[['point_2_x', 'point_2_y']].values)
            patch_l, patch_r = get_patch(img, points[0:1, :]), get_patch(img, points[1:, :])
            assert patch_l.shape == (32, 32), 'l error points {:.2f} {:.2f}'.format(points[0,0], points[0,1])
            assert patch_r.shape == (32, 32), 'r error points {:.2f} {:.2f}'.format(points[1,0], points[1,1])

            patch_l_list.append(patch_l)
            patch_r_list.append(patch_r)
            start_time = time.clock()
            hash_l = cmpHash(pHash(patch_l), pHash(patch_l_tgt))
            hash_r = cmpHash(pHash(patch_r), pHash(patch_r_tgt))
            # img_stack = np.vstack((np.hstack((patch_l,patch_r)), np.hstack((patch_l_tgt,patch_r_tgt))))
            # cv2.namedWindow('Hash', 0)
            # cv2.imshow('Hash', img_stack)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            speed_list.append(time.clock()-start_time)
            # if len(patch_l_list) > 10:
            #     patch_l_tgt_2 = patch_l_list[i-10]
            #     patch_r_tgt_2 = patch_r_list[i-10]
            # else:
            #     patch_l_tgt_2 = patch_l
            #     patch_r_tgt_2 = patch_r
            # hash_l_10 = cmpHash(pHash(patch_l), pHash(patch_l_tgt_2))
            # hash_r_10 = cmpHash(pHash(patch_r), pHash(patch_r_tgt_2))
            hash_l_10, hash_r_10 = 0, 0
            # print('hash value is l:{}, r:{}'.format(hash_l,hash_r))
            temp_df = pd.DataFrame({'index': img_idx, 'hash_l': hash_l, 'hash_r': hash_r,
                                    'fps': 1/np.mean(np.array(speed_list)),
                                    'hash_l_10': hash_l_10, 'hash_r_10': hash_r_10},
                                   index=[0], columns=df.columns)
            df = df.append(temp_df)
    df.to_csv('hash_analysis_track.csv')