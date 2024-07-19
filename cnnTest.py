import torch

import motion.motionDataMoc
from cnn.PrecisionNN import PrecisionNN
import numpy as np
from plot.resultPlot import *


def normalize_data(data):
    return (data - data.min(0)) / (data.max(0) - data.min(0))


# 1维卷积神经网络训练模型
if __name__ == '__main__':
    sample_len = 1
    # 各轴定位精度
    # precision = [0.1, 0.15, 0.2, 0.25, 0.3]
    loc_pre = [0.12, 0.13, 0.22, 0, 0]
    # 直线度
    straightness = [[0.000005, 0.0000035], [0.0000024, 0.0000033], [0.0000022, 0.0000041]]
    # straightness = [[0, 0], [0, 0], [0, 0]]
    # 角度偏差
    angleError = [[0.000003, 0.000002], [0.000004, 0.000002], [0.0000024, 0.000005]]
    # angleError = [[0, 0], [0, 0], [0, 0]]
    # 垂直度偏差
    verticalPre = [0.00001, 0.00002, 0.00001, 0., 0., 0., 0.]
    # verticalPre = [0.0, 0.0, 0.0, 0., 0., 0., 0.]
    precision = [loc_pre, straightness, angleError, verticalPre]
    # 各轴运动行程
    axis_range = [2000., 2000., 2000., 180., 180.]
    label = ["x", "y", "z", "S_XXY", "S_XXZ", "S_YYZ", "S_YYX", "S_ZZY", "S_ZZX", "A_XXY", "A_XXZ", "A_YYZ", "A_YYX",
             "A_ZZY", "A_ZZX", "VER_XY", "VER_ZX", "VER_ZY"]
    # 运动路线，为直线运动的终点坐标，取0-1，表示各轴的行程范围
    route = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1], [1, 1, 1]]
    data_x = []
    data_y = []
    all_data_x = []
    all_data_y = []
    # 生成各轴运动数据
    for i in range(len(route)):
        tmp_x, tmp_y = motion.motionDataMoc.generateDataCNN(50, sample_len, precision, route=route[i],
                                                            axis_range=axis_range)
        all_data_x.append(tmp_x)
        all_data_y.append(tmp_y)
    # 各轴运动路径、末端误差数据，整理为一个输入
    data_x = np.array(all_data_x).reshape(-1, len(all_data_x[0][0]))
    # 各轴误差数据作为标签数据，整理为一个数组
    data_y = np.array(all_data_y).reshape(-1, len(all_data_y[0][0]))
    # 生成一个路径的测试数据
    test_x, test_y = motion.motionDataMoc.generateDataCNN(50, sample_len, precision, route=[0.5, 1, 1],
                                                          axis_range=axis_range)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    # 最大迭代次数
    max_epochs = 50
    # 每批数据数量
    batch_size = 1
    # 学习率
    learning_rate = 0.0005
    # 初始化模型
    model = PrecisionNN([data_x, data_y], max_epochs, batch_size, learning_rate, loss_thres=0.005)
    # 开始模型训练
    model.run()
    train_pred = []
    test_pred = []
    # 训练数据转为tensor对象
    tensor_train_x = torch.tensor(data_x, dtype=torch.double).unsqueeze(1).unsqueeze(1)
    # 测试数据转为tensor对象
    tensor_test_x = torch.tensor(test_x, dtype=torch.double).unsqueeze(1).unsqueeze(1)
    for i in range(tensor_train_x.shape[0]):
        # 预测训练数据
        train_pred.append(model.model(tensor_train_x[i]).detach().numpy())
    for i in range(tensor_test_x.shape[0]):
        # 预测测试数据
        test_pred.append(model.model(tensor_test_x[i]).detach().numpy())
    # 转为numpy对象
    train_pred = np.array(train_pred)
    test_pred = np.array(test_pred)
    # train_pred = model.model(tensor_train_x)
    # test_pred = model.model(tensor_test_x)
    # 画出图形
    plot_cnn_result(data_x, data_y, train_pred, test_x, test_y, test_pred, label, axis_range)
    print(len(data_x), len(data_y))
