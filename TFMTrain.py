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
    data_len = 100
    # 多少组四线路径
    route_len = 10
    # 最大迭代次数
    max_epochs = 1
    # 每批数据数量
    batch_size = 1
    load_best = True
    # 学习率
    learning_rate = 0.0001
    test_route = [1, 1, 1]
    # 生成一个路径的测试数据
    # todo 误差增加负向误差
    test_error_param = [[4.7887194555963255, 0.4460618609979403, 0.478609241527035],
                        [0.6873725685235172, 2.9875751745178087, 0.9794735286526571],
                        [4.535475693069581, 2.9173525630956263, 0.5506782494905588],
                        [1.4049564502737972, 1.6158067832154326, 0.5234738589988253],
                        [0.7467095657561036, 4.515754506183082, 0.7608852455039745],
                        [3.00859321444696, 5.040013743165973, 0.45451975869615724],
                        [2.0979793289187607, 0.5504565321525232, 0.8295217960434513],
                        [0.3408292725935962, 3.430547548490598, 0.6385488419507832],
                        [5.925506277060648, 0.968058240941668, 0.7010201830502059],
                        [0.23145427669709195, 4.884802258319203, 0.9822079724307121],
                        [3.788902991626234, 5.354173413157307, 0.9937083523717993],
                        [3.9014983158102443, 0.9853261390752617, 0.2822661253420671],
                        [3.8930690035075006, 2.3945249711732965, 0.027491172062673708],
                        [2.5718760699285834, 2.4590918484517257, 0.043019414283773205],
                        [2.9537169160920196, 3.0866430925396813, 0.3261355358997603],
                        [0.48171559581933887, 4.798924831691007, 0.22651219118644284],
                        [5.094860945869729, 3.422991873304562, 0.9531220544221468],
                        [5.356315264062595, 4.130833225380868, 0.786862073839904]]

    data_x, data_y = motion.motionDataMoc.generateDataAll(data_len, route_len, sample_len, precision, axis_range,
                                                          magnification)
    test_x, test_y = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision, route=test_route,
                                                          axis_range=axis_range, pre_magnification=magnification,
                                                          error_param=test_error_param)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    #
    model_path = None
    weight_path = None
    if load_best:
        model_path = "./weights/TFMP_model.pt"
        weight_path = "./weights/TFMP_best.pt"
    # 初始化模型
    model = TFMPrecision([data_x, data_y], data_len=data_len, route_len=route_len, seq_length=data_len,
                         epoch=max_epochs, batch_size=batch_size, lr=learning_rate, loss_thres=0.005,
                         magnification=magnification, axis_range=axis_range, precision=precision, model_path=model_path,
                         weight_path=weight_path)
    # 开始模型训练
    model.run()
    # 训练数据转为tensor对象
    tensor_train_x = torch.tensor(data_x, dtype=torch.double, device=device)
    mask = torch.triu(torch.ones((data_len, data_len), device=device), diagonal=1).bool()
    train_pred = model.model(tensor_train_x).detach().cpu().numpy()
    train_pred = train_pred.reshape(-1, train_pred.shape[-1])
    # 四线的输入
    sample_x = []
    for i in range(len(route)):
        tmp_x, tmp_y = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision, route=route[i],
                                                            axis_range=axis_range, pre_magnification=magnification,
                                                            error_param=test_error_param)
        sample_x.append(tmp_x)
        if i == 0:
            target_y = tmp_y
    sample_x = np.array(sample_x)
    sample_x = np.transpose(sample_x, (1, 0, 2))
    sample_x = sample_x.reshape(data_len, len(route) * sample_x.shape[-1])
    # 预测整个机床的18项误差
    tensor_sample_x = torch.tensor(sample_x.tolist(), dtype=torch.double, device=device).unsqueeze(0)
    precision_grid = model.model(tensor_sample_x, mask).detach().cpu().numpy()
    precision_grid = precision_grid.reshape(-1, precision_grid.shape[-1])
    test_x_new, test_pred_new = motion.motionDataMoc.generate_pred_by_grid(precision_grid, data_len, sample_len,
                                                                           route=test_route,
                                                                           axis_range=axis_range,
                                                                           magnification=magnification)

    # 转为numpy对象
    train_pred = np.array(train_pred)
    test_pred = np.array(test_pred_new)
    avg = np.average(np.abs(test_pred - test_y))
    # 画出图形
    plot_cnn_result(data_x.reshape(-1, data_x.shape[-1]), data_y.reshape(-1, data_y.shape[-1]), train_pred, test_x,
                    test_y, test_pred, label, axis_range, magnification)
