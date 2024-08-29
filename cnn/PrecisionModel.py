import torch
import torch.nn as nn


# 精度辨识卷积神经网络模型
class PrecisionModel(nn.Module):
    k = 0
    max_channel = 64

    def __init__(self, input_size, output_size, batch_size):
        super(PrecisionModel, self).__init__()
        cMax = self.max_channel
        cMax_2 = int(cMax / 2)
        cMax_4 = int(cMax / 4)
        # 设定各个卷积层
        self.layer1 = nn.Sequential(nn.Conv1d(1, cMax_4, 3, padding=1), nn.BatchNorm1d(cMax_4), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv1d(cMax_4, cMax_2, 3, padding=1), nn.BatchNorm1d(cMax_2),
                                    nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv1d(cMax_2, cMax, 3, padding=1), nn.BatchNorm1d(cMax), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv1d(cMax, cMax_2, 3, padding=1), nn.BatchNorm1d(cMax_2),
                                    nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv1d(cMax_2, cMax_4, 3, padding=1), nn.BatchNorm1d(cMax_4),
                                    nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.Conv1d(cMax_2, cMax_2, 3, padding=1), nn.BatchNorm1d(cMax_2),
                                    nn.ReLU(inplace=True))
        # 设定全联结输出层
        self.last_layer = nn.Sequential(nn.Linear(cMax_2 * input_size, output_size))

    def forward(self, x):
        result1 = self.layer1(x)
        result = self.layer2(result1)
        result = self.layer3(result)
        result = self.layer4(result)
        result = self.layer5(result)
        # 设定残差层，直接将第1层输出与第5层输出结合，作为第六层输入
        result = torch.cat((result, result1), dim=1)
        result = self.layer6(result)
        result = result.view(-1)
        # result = self.last_layer(torch.cat((result, x.view(-1)[-1:]), dim=0))
        result = self.last_layer(result)
        return result
