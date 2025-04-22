import torch
import torch.nn as nn

from motion.motionCalculatorTorch import MotionCalculator
from motion.motionMatrixTorch import MotionMatrixTensor
from plot.resultPlot import array_to_precession_input


class MotionLoss(nn.Module):
    def __init__(self, magnification, axis_range, device):
        super(MotionLoss, self).__init__()
        self.magnification = magnification
        self.axis_range = axis_range
        MotionMatrixTensor.device = device

    def forward(self, inputs, targets):
        inputs_error = torch.zeros(inputs.shape[0], inputs.shape[1], 4)
        for i in range(inputs.shape[0]):
            for k in range(inputs.shape[1]):
                pre = inputs[i][k]
                padding_tensor = torch.zeros(2, dtype=pre.dtype)
                loc, straightness, angle = torch.concat((pre, padding_tensor)) / self.magnification[0], \
                                           pre[3:9].reshape(-1, 2) / self.magnification[1], \
                                           pre[9:18].reshape(-1, 3) / self.magnification[2]
                calculator = MotionCalculator()
                calculator.setAxisRange(self.axis_range)
                calculator.setPrecision(pLoc=loc / self.magnification[0],
                                        straightness=straightness / self.magnification[1],
                                        angleError=angle / self.magnification[2])
                calculator.setBiasACY(0)
                calculator.setL(0)
                error, m1, m2, m3, error_pjct = calculator.calculate(targets[i][k][4] / self.magnification[4],
                                                                     targets[i][k][5] / self.magnification[4],
                                                                     targets[i][k][6] / self.magnification[4], 0, 0)
                inputs_error[i][k][0] = error
                inputs_error[i][k][1] = m1
                inputs_error[i][k][2] = m2
                inputs_error[i][k][3] = m3
        # 计算均方误差
        diff = inputs_error - targets[:, :, 0:4]
        loss = torch.mean(diff ** 2)
        return loss
