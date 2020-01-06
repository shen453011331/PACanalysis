import cv2
import numpy as np
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
            return None

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
        vector_a = np.squeeze(P - P_rhorn)
        vector_b = np.squeeze(P_lhorn - P_rhorn)
        percent = vector_a.dot(vector_b)/np.square(np.linalg.norm(vector_b))
        vector_c = percent * vector_b
        distance = (0.5 - percent) * np.linalg.norm(vector_b)
        height = np.linalg.norm(vector_a - vector_c)
        return distance, height

    def getPolarLine(self, p_x, p_y, r_p_x, r_p_y):
        point_l = np.array([p_x, p_y])
        point_r = np.array([r_p_x, r_p_y])
        point_l_rm = self.removeDistortion(point_l, init_left)
        point_r_rm = self.removeDistortion(point_r, init_right)
        p_left = np.array([[point_l_rm[0], point_l_rm[1], 1]]).T
        p_right_T = np.array(([[point_r_rm[0], point_r_rm[1], 1]]))
        Ml1 = self.init_para_l.Matrix
        ml = np.zeros([3, 1], dtype='float')
        Mr1 = np.dot(self.init_para_r.Matrix, self.exte_para_1_2.R)
        mr = np.dot(self.init_para_r.Matrix, self.exte_para_1_2.T)
        temp = np.dot(Mr1, np.linalg.inv(Ml1))
        m = mr - np.dot(temp, ml)
        m_x = self.getTxMatrix(m)
        temp = np.dot(m_x, Mr1)
        F = np.dot(temp, np.linalg.inv(Ml1))
        line_r = np.dot(F, p_left)
        # test
        k = np.dot(p_right_T, line_r)
        # test P 3D
        P = self.reconstruct3D(point_l, point_r)
        p_left_temp = np.dot(Ml1, P)
        p_left_temp = p_left_temp / p_left_temp[2]
        P_r = np.dot(self.exte_para_1_2.R, P) + self.exte_para_1_2.T
        p_right_temp = np.dot(self.init_para_r.Matrix, P_r)
        p_right_temp = np.dot(Mr1, P) + mr
        p_right_temp = p_right_temp/p_right_temp[2]
        P_test = self.reconstruct3D(point_l, np.array([p_right_temp[0], p_right_temp[1]]))

        #
        A = np.zeros([4, 3], dtype='float')
        u1 = point_l_rm[0]
        v1 = point_l_rm[1]
        u2 = point_r_rm[0]
        v2 = point_r_rm[1]
        A[0,0] = Ml1[2, 0]*u1-Ml1[0,0]
        A[0,1] = u1*Ml1[2,1] - Ml1[0,1]
        A[0,2] = u1*Ml1[2,2] - Ml1[0,2]

        A[1,0] = Ml1[2, 0]*v1-Ml1[1,0]
        A[1, 1] = v1 * Ml1[2, 1] - Ml1[1, 1]
        A[1, 2] = v1 * Ml1[2, 2] - Ml1[1, 2]

        A[2, 0] = Mr1[2, 0] * u2 - Mr1[0, 0]
        A[2, 1] = u2 * Mr1[2, 1] - Mr1[0, 1]
        A[2, 2] = u2 * Mr1[2, 2] - Mr1[0, 2]
        A[3, 0] = Mr1[2, 0] * v2 - Mr1[1, 0]
        A[3, 1] = v2 * Mr1[2, 1] - Mr1[1, 1]
        A[3, 2] = v2 * Mr1[2, 2] - Mr1[1, 2]

        b = np.zeros([4, 1], dtype='float')
        b[0, 0] = ml[0] - u1* ml[2]
        b[1, 0] = ml[1] - v1* ml[2]
        b[2, 0] = mr[0] - u2 * mr[2]
        b[3, 0] = mr[1] - v2 * mr[2]
        temp_ata = np.dot(A.T, A)
        temp_atb = np.dot(A.T, b)
        P_temp = np.dot(np.linalg.inv(temp_ata), temp_atb)
        # 当前标定下对应于 left的right坐标为【1384，363】

        return line_r
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




