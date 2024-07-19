import numpy as np
from numpy import dot
from math import *
from enum import Enum


class RotateType(Enum):
    ANGLE = 0
    ABSOLUTE = 1


class MotionMatrix:
    def __init__(self, x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, g=0.0, rotateType=RotateType.ANGLE, dim=4):
        self.dim = dim
        self.matrix = np.diag([1.0] * self.dim)
        self.__rotateType = rotateType
        self.trans(x, y, z)
        self.rotateX(a, rotateType)
        self.rotateY(b, rotateType)
        self.rotateZ(g, rotateType)

    def print(self):
        print(self.matrix)

    def trans(self, x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, g=0.0, rotateType=RotateType.ANGLE):
        self.transX(x)
        self.transY(y)
        self.transZ(z)
        self.rotateX(a, rotateType)
        self.rotateY(b, rotateType)
        self.rotateZ(g, rotateType)
        return self

    def transX(self, x):
        if x != 0.0:
            self.matrix = dot(self.matrix, self._transMat(0, x))
        return self

    def transY(self, y):
        if y != 0.0:
            self.matrix = dot(self.matrix, self._transMat(1, y))
        return self

    def transZ(self, z):
        if z != 0.0:
            self.matrix = dot(self.matrix, self._transMat(2, z))
        return self

    def rotateX(self, ang, rotateType=RotateType.ANGLE):
        self.__rotateType = rotateType
        if ang != 0.0:
            self.matrix = dot(self.matrix, self._rotateMat(0, ang))
        return self

    def rotateY(self, ang, rotateType=RotateType.ANGLE):
        self.__rotateType = rotateType
        if ang != 0.0:
            self.matrix = dot(self.matrix, self._rotateMat(1, ang))
        return self

    def rotateZ(self, ang, rotateType=RotateType.ANGLE):
        self.__rotateType = rotateType
        if ang != 0.0:
            self.matrix = dot(self.matrix, self._rotateMat(2, ang))
        return self

    # 生成平移变换矩阵
    def _transMat(self, axis, dis):
        mat = np.diag([1.0] * self.dim)
        mat[axis][3] = dis
        return mat

    # 生成旋转变换矩阵
    def _rotateMat(self, axis, ang):
        mat = np.diag([1.0] * self.dim)
        if self.__rotateType == RotateType.ABSOLUTE:
            # 垂直度误差
            if axis == 0:
                mat[1][1] = 1
                mat[1][2] = -ang
                mat[2][1] = ang
                mat[2][2] = 1
            if axis == 1:
                mat[0][0] = 1
                mat[0][2] = ang
                mat[2][0] = -ang
                mat[2][2] = 1
            if axis == 2:
                mat[0][0] = 1
                mat[0][1] = -ang
                mat[1][0] = ang
                mat[1][1] = 1
        elif self.__rotateType == RotateType.ANGLE:
            # 按角度算
            ang = radians(ang)
            # ang = ang / 57.3
            if axis == 0:
                mat[1][1] = cos(ang)
                mat[1][2] = -sin(ang)
                mat[2][1] = sin(ang)
                mat[2][2] = cos(ang)
            if axis == 1:
                mat[0][0] = cos(ang)
                mat[0][2] = sin(ang)
                mat[2][0] = -sin(ang)
                mat[2][2] = cos(ang)
            if axis == 2:
                mat[0][0] = cos(ang)
                mat[0][1] = -sin(ang)
                mat[1][0] = sin(ang)
                mat[1][1] = cos(ang)

        return mat

    # 操作符重载，乘法重载
    def __mul__(self, right):
        ret = MotionMatrix()
        ret.matrix = self.matrix
        ret.__rotateType = RotateType.ANGLE
        ret.dim = self.dim
        if isinstance(right, np.matrix):
            ret.matrix = dot(ret.matrix, right)
        elif isinstance(right, MotionMatrix):
            ret.matrix = dot(ret.matrix, right.matrix)
        elif isinstance(right, list):
            ret.matrix = dot(ret.matrix, np.matrix(right))
        return ret
