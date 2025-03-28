import torch

import motion.motionDataMoc
from transformer.TFMPrecision import TFMPrecision
import numpy as np
from plot.resultPlot import *
import random

# transformer网络训练模型
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sample_len = 0
    # 各轴定位精度
    # precision = [0.1, 0.15, 0.2, 0.25, 0.3]
    loc_pre = [0.12, 0.13, 0.11, 0, 0]
    # 直线度
    straightness = [[0.000005, 0.0000035], [0.0000024, 0.0000033], [0.0000022, 0.0000041]]
    # straightness = [[0.05, 0.035], [0.024, 0.033], [0.022, 0.041]]
    # straightness = [[0, 0], [0, 0], [0, 0]]
    # 角度偏差
    angleError = [[0.000003, 0.000002, 0.000002], [0.000004, 0.000002, 0.000002], [0.0000024, 0.000005, 0.0000031]]
    # angleError = [[0.03, 0.02, 0.02], [0.04, 0.02, 0.02], [0.024, 0.05, 0.031]]
    # angleError = [[0, 0], [0, 0], [0, 0]]
    # 垂直度偏差
    # verticalPre = [0.00001, 0.00002, 0.00001, 0., 0., 0., 0.]
    # verticalPre = [0.0, 0.0, 0.0, 0., 0., 0., 0.]
    precision = [loc_pre, straightness, angleError]
    # 精度放大倍率，行程缩小倍率
    magnification = [5, 100000, 100000, 10000, 0.0001]
    # magnification = [1, 1, 100000, 10000, 0.0001]
    # 各轴运动行程
    axis_range = [2000., 2000., 2000., 180., 180.]
    label = ["x", "y", "z", "S_XXY", "S_XXZ", "S_YYZ", "S_YYX", "S_ZZY", "S_ZZX", "A_XA", "A_XB", "A_XC", "A_YA",
             "A_YB", "A_YC", "A_ZA", "A_ZB", "A_ZC"]
    # 运动路线，为直线运动的终点坐标，取0-1，表示各轴的行程范围
    route = [[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]
    # route = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1], [1, 1, 1]]
    data_x = []
    data_y = []
    all_data_x = []
    all_data_y = []
    data_len = 100
    # 多少组四线路径
    route_len = 20
    # 最大迭代次数
    max_epochs = 3000
    # 每批数据数量
    batch_size = 4
    # 学习率
    learning_rate = 0.0001
    test_route = [1, 0.7, 0.5]
    # 生成一个路径的测试数据
    test_error_param = [[-2.626736262022589, 3.9816823239449013, 0.4475779332261868],
                        [4.898369153866513, 0.6535184026344443, 0.3959563017327412],
                        [0.9533893812664391, 4.321387650379188, 0.02204032468391437],
                        [0.24440702204519904, -3.958786536533847, 0.6583649830759168],
                        [3.5905037616206474, -1.9612116414839114, 0.39881849320093066],
                        [5.035059919859897, -3.3730870998153963, 0.24439578721144561],
                        [-1.3507938693774302, -1.3721232074566159, 0.07062618877740445],
                        [-5.51405717986294, 0.9911795643678536, 0.8182693704870628],
                        [-2.4276056326006232, -0.01938183378397973, 0.6143838486206817],
                        [3.232230008214886, -2.3719314887049263, 0.2296585006296279],
                        [-0.8467920258479993, -4.27827754416902, 0.959136283574746],
                        [-4.039960769419381, -4.648571136021942, 0.7479244636780055],
                        [-1.0923572392653118, -4.966197291891709, 0.6492033619068038],
                        [-2.031126625296771, 4.465203600405606, 0.8525929513569045],
                        [-3.0316317013854395, -5.873128889484105, 0.6901978943595463],
                        [-1.9320195212275568, 5.326115305879207, 0.7308464096223587],
                        [-5.451774079287787, -2.7198500874180915, 0.5603441309229098],
                        [4.5396633186930435, 4.929125528584144, 0.7249377525195019]]
    # 生成各轴运动数据
    for k in range(route_len):
        # 18项误差的随机值
        # loc_pre_random = []
        # straightness_random = []
        # angle_error_random = []
        # precision_random = [loc_pre_random, straightness_random, angle_error_random]
        precision_random = precision
        # 18项误差线性变化随机值
        error_params_random = [[(1 if random.random() - 0.5 > 0 else -1) * random.random() * 6,
                                (1 if random.random() - 0.5 > 0 else -1) * random.random() * 6, random.random()]
                               for n in range(18)]
        for i in range(len(route)):
            tmp_x, tmp_y = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision_random, route=route[i],
                                                                axis_range=axis_range, pre_magnification=magnification,
                                                                error_param=error_params_random)
            all_data_x.append(tmp_x)
            all_data_y.append(tmp_y)
    # 各轴运动路径、末端误差数据，整理为一个输入
    data_x = np.array(all_data_x)
    # 各轴误差数据作为标签数据，整理为一个数组
    data_y = np.array(all_data_y)
    test_x, test_y = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision, route=test_route,
                                                          axis_range=axis_range, pre_magnification=magnification,
                                                          error_param=test_error_param)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    # 初始化模型
    model = TFMPrecision([data_x, data_y], seq_length=data_len, epoch=max_epochs, batch_size=batch_size,
                         lr=learning_rate, loss_thres=0.005)
    # 开始模型训练
    model.run()
    # 训练数据转为tensor对象
    tensor_train_x = torch.tensor(data_x, dtype=torch.double, device=device)
    # 体对角线的输入
    sample_x, _y = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision, route=[1, 1, 1],
                                                        axis_range=axis_range, pre_magnification=magnification,
                                                        error_param=test_error_param)
    # 体对角线转换为tensor对象
    tensor_sample_x = torch.tensor(sample_x, dtype=torch.double, device=device).unsqueeze(0)
    train_pred = model.model(tensor_train_x).detach().cpu().numpy()
    train_pred = train_pred.reshape(-1, train_pred.shape[-1])
    # 预测整个机床的18项误差
    precision_grid = model.model(tensor_sample_x).detach().cpu().numpy()
    precision_grid = precision_grid.reshape(-1, precision_grid.shape[-1])
    test_x_new, test_pred_new = motion.motionDataMoc.generate_pred_by_grid(precision_grid, data_len, sample_len,
                                                                           route=test_route,
                                                                           axis_range=axis_range,
                                                                           magnification=magnification)

    # 转为numpy对象
    train_pred = np.array(train_pred)
    test_pred = np.array(test_pred_new)
    # 画出图形
    plot_cnn_result(data_x.reshape(-1, data_x.shape[-1]), data_y.reshape(-1, data_y.shape[-1]), train_pred, test_x,
                    test_y, test_pred, label, axis_range, magnification)
