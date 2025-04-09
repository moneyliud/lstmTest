import numpy as np
from numpy import dot
from math import *
from enum import Enum
import torch
from motion.motionMatrix import MotionMatrix
from motion.motionMatrix import RotateType


class MotionMatrixTensor(MotionMatrix):
    device = "cpu"

    def __init__(self, x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, g=0.0, rotateType=RotateType.ANGLE, dim=4):
        super().__init__(x, y, z, a, b, g, rotateType, dim, device=MotionMatrixTensor.device)
        # self.matrix = torch.tensor(self.matrix, device=MotionMatrixTensor.device)

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
            self.matrix = torch.matmul(self.matrix, self._transMat(0, x))
        return self

    def transY(self, y):
        if y != 0.0:
            self.matrix = torch.matmul(self.matrix, self._transMat(1, y))
        return self

    def transZ(self, z):
        if z != 0.0:
            self.matrix = torch.matmul(self.matrix, self._transMat(2, z))
        return self

    def rotateX(self, ang, rotateType=RotateType.ANGLE):
        self.__rotateType = rotateType
        if ang != 0.0:
            self.matrix = torch.matmul(self.matrix, self._rotateMat(0, ang))
        return self

    def rotateY(self, ang, rotateType=RotateType.ANGLE):
        self.__rotateType = rotateType
        if ang != 0.0:
            self.matrix = torch.matmul(self.matrix, self._rotateMat(1, ang))
        return self

    def rotateZ(self, ang, rotateType=RotateType.ANGLE):
        self.__rotateType = rotateType
        if ang != 0.0:
            self.matrix = torch.matmul(self.matrix, self._rotateMat(2, ang))
        return self

    # 生成平移变换矩阵
    def _transMat(self, axis, dis):
        mat = super(MotionMatrixTensor, self)._transMat(axis, dis)
        mat_tensor = torch.tensor(mat, device=MotionMatrixTensor.device, requires_grad=True)
        return mat_tensor

    # 生成旋转变换矩阵
    def _rotateMat(self, axis, ang):
        mat = super(MotionMatrixTensor, self)._rotateMat(axis, ang)
        mat_tensor = torch.tensor(mat, device=MotionMatrixTensor.device, requires_grad=True)
        return mat_tensor

    # 操作符重载，乘法重载
    def __mul__(self, right):
        ret = MotionMatrixTensor()
        ret.matrix = self.matrix
        ret.__rotateType = RotateType.ANGLE
        ret.dim = self.dim
        if isinstance(right, torch.Tensor):
            ret.matrix = torch.matmul(ret.matrix, right)
        elif isinstance(right, MotionMatrixTensor):
            ret.matrix = torch.matmul(ret.matrix, right.matrix)
        elif isinstance(right, list):
            ret.matrix = torch.matmul(ret.matrix,
                                      torch.tensor(right, device=MotionMatrixTensor.device, requires_grad=True))
        return ret
