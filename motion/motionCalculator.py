import numpy as np
from motion.motionMatrix import MotionMatrix
from motion.motionMatrix import RotateType
import math
from random import random

TRANS_PRECISION = 0.01
SP_PRECISION = 0.005
ROTATE_PRECISION = 0.00001
VERTICAL_PRECISION = 0.000001

_x = 0
_y = 1
_z = 2
_c = 3
_a = 4


def _randomTrans(p=TRANS_PRECISION):
    return random() * p
    # return 0.8 * p
    # return p


def _randomRotate(p=ROTATE_PRECISION):
    return (random() * 2 - 1) * p
    # a = 1
    # if random() < 0.5:
    #     a = -1
    # return a * p
    # return p
    # return (0.8 * 2 - 1) * p


def _binaryRotate(p, flag):
    if flag > 0:
        return p
    else:
        return -p


class MotionCalculator:
    def __init__(self):
        self.__L = 0.0
        self.__LMat = None
        self.__B_ACY = 0.0
        self.__BMat = None
        # 运动范围 x,y,z,c,a
        self.__axis_range = [0.0, 0.0, 0.0, 0.0, 0.0]
        # 定位精度 x,y,z,c,a
        self.__p_loc = [0.0, 0.0, 0.0, 0.0, 0.0]
        # 重复定位精度
        self.__p_loc_re = [0.0, 0.0, 0.0, 0.0, 0.0]
        # 直线度
        self.__straightness = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        # 角度误差
        self.__angleError = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # 垂直度误差
        self.__verticalPre = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.__p_trans = TRANS_PRECISION
        self.__p_rotate = ROTATE_PRECISION
        self.__p_vertical = VERTICAL_PRECISION
        self.__pIdeal = None
        self.__pActual = None
        self.__pError = None

    def calculate(self, x, y, z, c, a):
        # Pideal理想值
        # 平移x,y,z,旋转c偏置B，旋转a，偏置L
        self.__pIdeal = MotionMatrix(x, y, z).rotateZ(c) * self.__BMat * MotionMatrix().rotateX(a)
        # self.__pIdeal = MotionMatrix(x, y, z).rotateZ(c) * MotionMatrix().rotateX(a)
        self.__pIdeal = self.__pIdeal * self.__LMat
        # x静态误差
        # 平移x带随机误差，直线度随机y,z,随机旋转δα、δβ、δγ
        e_static_x = MotionMatrix(self.__randomError(x, _x), _randomTrans(self.__straightness[_x][0]),
                                  _randomTrans(self.__straightness[_x][1]),
                                  # 倾斜
                                  _randomRotate(self.__angleError[_x][0]),
                                  # _randomRotate(),
                                  # 俯仰
                                  _randomRotate(self.__angleError[_x][1]),
                                  # _randomRotate(),
                                  _randomRotate(self.__angleError[_x][2]), RotateType.ABSOLUTE)

        # y静态误差
        # 平移y带随机误差，直线度随机x,z,随机旋转δα、δβ、δγ
        e_static_y = MotionMatrix(_randomTrans(self.__straightness[_y][0]), self.__randomError(y, _y),
                                  _randomTrans(self.__straightness[_y][1]),
                                  # 俯仰
                                  _randomRotate(self.__angleError[_y][0]),
                                  # _randomRotate(),
                                  # 倾斜
                                  _randomRotate(self.__angleError[_y][1]),
                                  # _randomRotate(),
                                  _randomRotate(self.__angleError[_y][2]),
                                  RotateType.ABSOLUTE)

        # z静态误差
        # 平移z带随机误差，直线度随机x,y,随机旋转δα、δβ、δγ
        e_static_z = MotionMatrix(_randomTrans(self.__straightness[_z][0]), _randomTrans(self.__straightness[_z][1]),
                                  self.__randomError(z, _z),
                                  # 在ZY平面内旋转
                                  _randomRotate(self.__angleError[_z][0]),
                                  # _randomRotate(),
                                  # 在ZX平面内旋转
                                  _randomRotate(self.__angleError[_z][1]),
                                  # _randomRotate(),
                                  _randomRotate(self.__angleError[_z][2]), RotateType.ABSOLUTE)

        # c静态误差
        # 按c定位精度、重复定位精度，随机旋转δα、δβ、δγ
        e_static_c = MotionMatrix(0.0, 0.0, 0.0,
                                  self.__randomError(c, _c), self.__randomError(c, _c), self.__randomError(c, _c))
        # a静态误差
        # 按a定位精度、重复定位精度，随机旋转δα、δβ、δγ
        e_static_a = MotionMatrix(0.0, 0.0, 0.0,
                                  self.__randomError(a, _a), self.__randomError(a, _a), self.__randomError(a, _a))
        # SP静态误差
        # x按TRANS_PRECISION随机平移，y按SP_PRECISION随机平移，z不动，随机旋转δα、δβ、δγ
        e_static_sp = MotionMatrix(_randomTrans(), _randomTrans(SP_PRECISION), 0.0, _randomRotate(), _randomRotate(),
                                   _randomRotate(), RotateType.ABSOLUTE)

        # 随机旋转
        rd = _randomRotate
        # vp = VERTICAL_PRECISION
        # 绝对位置
        rtype = RotateType.ABSOLUTE
        # 垂直度误差xy,zx,zy,cx,cy,cb,ca
        e_v_xy = MotionMatrix().rotateZ(rd(self.__verticalPre[0]), rtype)
        e_v_zx = MotionMatrix().rotateY(rd(self.__verticalPre[1]), rtype)
        e_v_zy = MotionMatrix().rotateX(rd(self.__verticalPre[2]), rtype)
        e_v_cx = MotionMatrix().rotateY(rd(self.__verticalPre[3]), rtype)
        e_v_cy = MotionMatrix().rotateX(rd(self.__verticalPre[4]), rtype)
        e_v_cb = MotionMatrix().rotateX(rd(self.__verticalPre[5]), rtype)
        e_v_ca = MotionMatrix().rotateY(rd(self.__verticalPre[6]), rtype)

        # e_v_xy = MotionMatrix().rotateZ(rd(vp), rtype)
        # e_v_zx = MotionMatrix().rotateY(rd(vp), rtype)
        # e_v_zy = MotionMatrix().rotateX(rd(vp), rtype)
        # e_v_cx = MotionMatrix().rotateY(rd(vp), rtype)
        # e_v_cy = MotionMatrix().rotateX(rd(vp), rtype)
        # e_v_cb = MotionMatrix().rotateX(rd(VERTICAL_PRECISION), rtype)
        # e_v_ca = MotionMatrix().rotateY(rd(vp), rtype)

        # 实际精度
        self.__pActual = MotionMatrix().transX(x) * e_static_x * e_v_xy * \
                         MotionMatrix().transY(y) * e_static_y * e_v_zx * e_v_zy * \
                         MotionMatrix().transZ(z) * e_static_z * e_v_cx * e_v_cy * \
                         MotionMatrix().rotateZ(c) * e_static_c * self.__BMat * e_v_cb * e_v_ca * \
                         MotionMatrix().rotateX(a) * e_static_a * MotionMatrix() * e_static_sp

        # self.__pActual = MotionMatrix().transX(x) * e_static_x * e_v_xy * \
        #                  MotionMatrix().transY(y) * e_static_y * e_v_zx * e_v_zy * \
        #                  MotionMatrix().transZ(z) * e_static_z * e_v_cx * e_v_cy * \
        #                  MotionMatrix().rotateZ(c) * e_static_c * e_v_cb * e_v_ca * \
        #                  MotionMatrix().rotateX(a) * e_static_a * MotionMatrix() * e_static_sp
        self.__pActual = self.__pActual * self.__LMat
        # print((MotionMatrix().rotateX(c)).matrix)
        # 误差
        self.__pError = self.__pActual.matrix - self.__pIdeal.matrix
        p_error_list = self.__pError.tolist()
        m1 = p_error_list[0][0]
        m2 = p_error_list[1][0]
        m3 = p_error_list[2][0]

        e_dis = math.sqrt(m1 * m1 + m2 * m2 + m3 * m3)
        # print("loc_z")
        # print(self.__p_loc[_z])
        # print(e_dis)
        # todo error不要用m1,m2,m3的模来计算, 改为误差向量在理论向量方向上的投影长度误差
        return e_dis, m1, m2, m3

    # 偏置
    def setBiasACY(self, bias):
        self.__B_ACY = bias
        self.__BMat = MotionMatrix(0.0, bias)

    # 摆长
    def setL(self, length):
        self.__L = length
        self.__LMat = np.matrix([[0], [0], [-self.__L], [1]])

    # 轴运动范围
    def setAxisRange(self, axisRange):
        if len(self.__axis_range) != len(axisRange):
            raise Exception("运动范围数组不正确")
        self.__axis_range = axisRange

    # 定位精度、重复定位精度、直线度赋值
    def setPrecision(self, pLoc=None, pLocRe=None, straightness=None, angleError=None, verticalPre=None):
        if pLoc is not None:
            self.__p_loc = pLoc
        if pLocRe is not None:
            self.__p_loc_re = pLocRe
        if straightness is not None:
            self.__straightness = straightness
        if angleError is not None:
            self.__angleError = angleError
        if verticalPre is not None:
            self.__verticalPre = verticalPre
        # if len(self.__p_loc) != len(pLoc) or len(self.__p_loc_re) != len(pLocRe):
        #     raise Exception("精度数组不正确")

    # 按定位精度、重复定位精度、轴行程范围计算随机误差
    def __randomError(self, coord, axis):
        # ret = self.__p_loc[axis] / self.__axis_range[axis] * coord + _randomRotate(self.__p_loc_re[axis])
        # if axis == _x:
        #     print("x", ret)
        # return ret
        ret = self.__p_loc[axis] / self.__axis_range[axis] * coord + _randomRotate(self.__p_loc_re[axis])
        return ret
        # if axis == _z:
        #     ret = self.__p_loc[axis] / self.__axis_range[axis] * coord - _randomRotate(self.__p_loc_re[axis])
        # else:
        #     ret = self.__p_loc[axis] / self.__axis_range[axis] * coord + _randomRotate(self.__p_loc_re[axis])
        # return ret
