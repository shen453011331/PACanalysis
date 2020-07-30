import cv2
import numpy as np
import math


class init_para(object):
    def __init__(self, fx, fy, u0, v0, k1, k2):
        self.fx = fx
        self.fy = fy
        self.u0 = u0
        self.v0 = v0
        self.k1 = k1
        self.k2 = k2
        self.Matrix= np.array([[fx,0,u0], [0,fy,v0], [0,0,1]], dtype='float')


class exte_para(object):
    def __init__(self, R, T):
        self.R = np.array(R)
        self.T = np.array(T).T


init_left = init_para(4226.17, 4226.39,
                      1177.99, 932.03-500,
                      0.384, 0.0)
init_right = init_para(4270.22, 4278.11,
                       1156.43, 943.39-500,
                       0.323, 0.0)
# init_left = init_para(4324.736356157503, 4325.040423297528,
#                       1110.831942266570, 820.527904092132-600,
#                       -0.029480163150041, 0.0)
# init_right = init_para(4290.1066528232490, 4275.403439917222,
#                       1142.8072052916930, 846.6912882988600-600,
#                       -0.078652525529030, 0.0)
#
# RT_left_to_right = exte_para([[0.753836197549067, -0.146765479367138, 0.640461459676457],
#                               [0.158636347105862, 0.986552387431344, 0.039356018985954],
#                               [-0.637624887094029, 0.071932474722121, 0.766981239952242]],
#                              [[-888.4407489968044, -85.28604100183690, 333.83150733701460]])
# RT_ground = exte_para([[0.952645147114698, -0.077925509185021, 0.293929989448288],
#                        [0.201626132704495, 0.885440647543616, -0.418738298091029],
#                        [-0.227627165095568, 0.458172974645818, 0.859222554996526]],
#                       [[-317.653104213039, 606.059912250492, -1086.211322511339]])
v_left_right = [-0.0013, 0.2777, 0.1130]
T_left_right = [[-973.25, -34.92, 95.22]]
R,_ = cv2.Rodrigues(np.array(v_left_right))
R_left_right = R.tolist()
RT_left_to_right = exte_para(R_left_right, T_left_right)
v_ground = [0.0133, 0.3476, -0.0507]
T_ground = [[-770.96, 4328.53, 0.01]]
R,_ = cv2.Rodrigues(np.array(v_ground))
R_ground = R.tolist()
RT_ground = exte_para(R_ground, T_ground)


class Process3D(object):
    def __init__(self):
        self.init_para_l = init_left
        self.init_para_r = init_right
        self.exte_para_1_2 = RT_left_to_right
        self.exte_para_ground = RT_ground

    def reconstruct3D(self, point_l, point_r):
        # point is a numpy with 1*2 [x,y]
        if point_l[0] > 0 and point_r[0] > 0:
            # 用于排除掉已知得定位错误的点
            point_l_rm = self.removeDistortion(point_l, init_left)
            point_r_rm = self.removeDistortion(point_r, init_right)
            P = self.construct3D_left_coor(point_l_rm, point_r_rm)
            P = self.change_coor_to_ground(P)
            return P
        else:
            return np.array([[0, 0, 0]]).T

    def removeDistortion(self, point, init_para_cam):
        xd = (point[0] - init_para_cam.u0) / init_para_cam.fx
        yd = (point[1] - init_para_cam.v0) / init_para_cam.fy
        xu = xd
        yu = yd
        for i in range(10):
            r2 = xu * xu + yu * yu
            rd = 1 + init_para_cam.k1 * r2 + init_para_cam.k2 * r2 * r2
            xu = xd / rd
            yu = yd / rd
        point_rm = np.array([init_para_cam.fx * xu + init_para_cam.u0, init_para_cam.fy * yu + init_para_cam.v0])
        return point_rm

    def construct3D_left_coor(self, point_l, point_r):
        du_lft = point_l[0] - self.init_para_l.u0
        dv_lft = point_l[1] - self.init_para_l.v0
        du_rgt = point_r[0] - self.init_para_r.u0
        dv_rgt = point_r[1] - self.init_para_r.v0
        tmp1 = self.init_para_l.fx * self.init_para_l.fy * \
               (du_rgt * self.exte_para_1_2.T[2] -
                  self.init_para_r.fx * self.exte_para_1_2.T[0])
        tmp2 = self.init_para_l.fy * (self.init_para_r.fx * self.exte_para_1_2.R[0, 0]
                                      - du_rgt * self.exte_para_1_2.R[2, 0]) * du_lft + \
               self.init_para_l.fx * (self.init_para_r.fx * self.exte_para_1_2.R[0, 1] -
                                       du_rgt * self.exte_para_1_2.R[2, 1]) + \
               self.init_para_l.fx * self.init_para_l.fy * \
               (self.init_para_r.fx * self.exte_para_1_2.R[0, 2] - du_rgt * self.exte_para_1_2.R[2, 2])
        zc = tmp1 / tmp2
        xc = (du_lft * zc) / self.init_para_l.fx
        yc = (dv_lft * zc) / self.init_para_l.fy


        P = np.array([xc, yc, zc])

        # test
        P_temp = np.dot(self.exte_para_1_2.R, P) + self.exte_para_1_2.T
        k1 = P_temp[0, 0] / P_temp[1, 0]
        k2 = du_rgt/dv_rgt

        k1_l = P[0,0]/ P[1,0]
        k2_l = du_lft/dv_lft
        return P

    def change_coor_to_ground(self, P):
        temp = np.dot(self.exte_para_ground.R, P)
        P = temp + self.exte_para_ground.T
        return P

    def calculateShift(self, P, P_lhorn, P_rhorn):
        # each points is a 3D points with numpy 1*3 [x,y,z]
        # theta angle of pantograph compare to herizonal
        vector_a = np.squeeze(P - P_rhorn)
        vector_b = np.squeeze(P_lhorn - P_rhorn)
        theta = math.atan2(vector_b[2], vector_b[0])
        percent = vector_a.dot(vector_b)/np.square(np.linalg.norm(vector_b))
        vector_c = percent * vector_b
        distance = (0.5 - percent) * np.linalg.norm(vector_b)
        height = np.linalg.norm(vector_a - vector_c)
        length = np.linalg.norm(vector_b)
        return distance, height, length, theta

    def getPolarLine(self, point_l, point_r):
        # point_l = np.array([p_x, p_y])
        point_l_rm = self.removeDistortion(point_l, init_left)
        p_left = np.array([[point_l_rm[0], point_l_rm[1], 1]]).T
        E = np.dot(self.getTxMatrix(self.exte_para_1_2.T), self.exte_para_1_2.R)
        K_r_minusT = np.linalg.inv(self.init_para_r.Matrix).T
        K_l_minus = np.linalg.inv(self.init_para_l.Matrix)
        F = np.dot(K_r_minusT, np.dot(E, K_l_minus))
        line_r = np.dot(F, p_left)

        # test
        # point_r = np.array([r_p_x, r_p_y])
        point_r_rm = self.removeDistortion(point_r, init_right)
        p_right_T = np.array(([[point_r_rm[0], point_r_rm[1], 1]]))
        k = np.dot(p_right_T, line_r)
        return line_r, k
        pass

    def getTxMatrix(self, t):
        # t is a 3*1 numpy vector
        t_x = np.zeros([3, 3], dtype='float')
        t_x[0, 1] = -t[2]
        t_x[0, 2] = t[1]
        t_x[1, 0] = t[2]
        t_x[1, 2] = -t[0]
        t_x[2, 0] = -t[1]
        t_x[2, 1] = t[0]
        return t_x




