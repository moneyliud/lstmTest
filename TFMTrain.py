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
    # loc_pre = [0.0, -0.0, 0.0, 0, 0]
    loc_pre = [0.12, -0.13, 0.11, 0, 0]
    # 直线度
    straightness = [[0.000005, -0.0000035], [-0.0000024, 0.0000033], [0.0000022, 0.0000041]]
    # straightness = [[0.05, 0.035], [0.024, 0.033], [0.022, 0.041]]
    # straightness = [[0, 0], [0, 0], [0, 0]]
    # 角度偏差
    angleError = [[0.000003, -0.000002, 0.000002], [0.000004, -0.000002, 0.000002], [-0.0000024, 0.000005, 0.0000031]]
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
    route = [[[0, 0, 0], [1, 1, 1]],
             [[0, 0, 0], [1, 0, 1]],
             [[0, 0, 0], [1, 1, 0]],
             [[0, 0, 0], [0, 1, 1]]]
    # route = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1], [1, 1, 1]]
    data_len = 100
    # 多少组四线路径
    route_len = 10
    # 最大迭代次数
    max_epochs = 200
    # 每批数据数量
    batch_size = 5
    load_best = True
    # 学习率
    learning_rate = 0.0002
    test_route = [[0, 0, 0], [1, 1, 1]]
    # 生成一个路径的测试数据
    test_error_param = [[5.192216764043185, 0.2617623586880524, 0.7588709747758215],
                        [-1.4290295470882306, -4.547730223217397, 0.4450699781137264],
                        [-3.9988284165462975, -4.695433768922194, 0.5694013306487599],
                        [5.797518076409893, -1.677077773142488, 0.2943403735099307],
                        [-2.7230039086054374, -0.3210501483310497, 0.9520080988055777],
                        [0.29041837810345106, -4.408891321092849, 0.746581009165349],
                        [-4.6072399701975915, 5.777708678994299, 0.2128682111676471],
                        [0.199459930106195, 5.219512134667823, 0.3650560484759472],
                        [-4.015008944027057, -4.987407252634625, 0.7271152856238464],
                        [-0.3010420040739392, -1.7338629273112747, 0.7887697444979137],
                        [-2.6886757782153117, -1.3602215246889653, 0.7504783552791788],
                        [-1.0347941535390592, 4.868790021853812, 0.36076933713507364],
                        [-4.100375491051398, 2.990065915463194, 0.6231219635141084],
                        [-0.4667005234813961, 1.1625127293042012, 0.5680719997341827],
                        [1.0461739313853164, 0.7954077689276728, 0.08202414377159173],
                        [-3.217261691105591, 0.06415096695995937, 0.6182923489750511],
                        [-3.705285190345603, -5.074995507147135, 0.49576772566349103],
                        [3.5245861908817053, -5.777751854551074, 0.6649989686033323]]

    route_data13, theory_route_data13, solve_result13, precision_target13 = motion.motionDataMoc.generateData13Line(
        data_len, sample_len, precision, axis_range=axis_range, pre_magnification=magnification,
        error_param=test_error_param)
    data_x, data_y = motion.motionDataMoc.generateDataAll(data_len, route_len, sample_len, precision, axis_range,
                                                          magnification, error_param=test_error_param)
    test_x, test_y, _tr = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision, route=test_route,
                                                               axis_range=axis_range, pre_magnification=magnification,
                                                               error_param=test_error_param)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # 画出图形
    plot_cnn_result(data_x.reshape(-1, data_x.shape[-1]), data_y.reshape(-1, data_y.shape[-1]), None, test_x,
                    test_y, solve_result13, label, axis_range, magnification, dir_path="image13Line")

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
        tmp_x, tmp_y, _tr = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision, route=route[i],
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
