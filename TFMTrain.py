import torch

import motion.motionDataMoc
from transformer.TFMPrecision import TFMPrecision
import numpy as np
from plot.resultPlot import *
import random
import time

# transformer网络训练模型
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sample_len = 0
    # 刀长
    L = 0
    # 各轴定位精度
    # precision = [0.1, 0.15, 0.2, 0.25, 0.3]
    # loc_pre = [0.0, -0.0, 0.0, 0, 0]
    loc_pre = [0.08, 0.051, -0.05, 0, 0]
    # 直线度
    # straightness = [[0.000025, -0.000031], [-0.000024, 0.000033], [0.000012, 0.000016]]
    straightness = [[0.000005, -0.000001], [-0.000004, 0.000003], [0.000002, 0.000006]]
    # 角度偏差
    angleError = [[0.000018, -0.00002, 0.000008], [0.000028, -0.000012, 0.00002], [0.000011, 0.000017, 0.000024]]
    # angleError = [[0.03, 0.02, 0.02], [0.04, 0.02, 0.02], [0.024, 0.05, 0.031]]
    # angleError = [[0., -0., 0.], [0.00004, -0.00002, 0.00002], [-0.0, 0.0, 0]]
    # 垂直度偏差
    # verticalPre = [0.00001, 0.00002, 0.00001, 0., 0., 0., 0.]
    # verticalPre = [0.0, 0.0, 0.0, 0., 0., 0., 0.]
    precision = [loc_pre, straightness, angleError]
    # 精度放大倍率，行程缩小倍率
    magnification = [5, 10000, 10000, 10000, 0.0001]
    # magnification = [1, 1, 100000, 10000, 0.0001]
    # 各轴运动行程
    axis_range = [2000., 2000., 2000., 180., 180.]
    label = ["$\delta X(\mathit{x})$", "$\delta Y(y)$", "$\delta Z(z)$",
             "$\delta Y(\mathit{x})$", "$\delta Z(\mathit{x})$",
             "$\delta Z(\mathit{y})$", "$\delta X(\mathit{y})$",
             "$\delta Y(\mathit{z})$", "$\delta X(\mathit{z})$",
             "$\delta \\alpha(\mathit{x})$", "$\delta \\beta(\mathit{x})$", "$\delta \\gamma(\mathit{x})$",
             "$\delta \\alpha(\mathit{y})$", "$\delta \\beta(\mathit{y})$", "$\delta \\gamma(\mathit{y})$",
             "$\delta \\alpha(\mathit{z})$", "$\delta \\beta(\mathit{z})$", "$\delta \\gamma(\mathit{z})$"]
    # 运动路线，为直线运动的终点坐标，取0-1，表示各轴的行程范围
    route = [[[0, 0, 0], [1, 1, 1]],
             [[0, 0, 0], [1, 0, 1]],
             [[0, 0, 0], [1, 1, 0]],
             [[0, 0, 0], [0, 1, 1]]]
    # route = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1], [1, 1, 1]]
    data_len = 40
    # 多少组四线路径
    route_len = 30
    # 最大迭代次数
    max_epochs = 100
    # 每批数据数量
    batch_size = 5
    load_best = True
    # 学习率
    learning_rate = 0.0002
    test_route = [[0, 0, 0], [1, 1, 1]]
    # 生成一个路径的测试数据
    test_error_param = [[2.192216764043185, 0.2617623586880524, 0.6588709747758215],
                        [-1.4290295470882306, -4.547730223217397, 0.4450699781137264],
                        [3.9988284165462975, 1.695433768922194, 0.4],
                        [1.178956203443989e-05, 2.059362224365455e-06, 0.4518755534305855],
                        [-2.271810302613115e-05, -3.786735616103811e-05, 0.26239433992585426],
                        [-2.780283293819153e-05, -1.4505684959253678e-05, 0.8432095919877094],
                        [4.095482930554621e-05, -1.92134835733238e-05, 0.6519152919184402],
                        [8.496095888274791e-06, 1.0077434424164404e-05, 0.12849043981522945],
                        [-3.77811841379065e-06, -2.517291154461439e-06, 0.4050594493584998],
                        [1.7011455515135813e-05, -1.678518859535065e-06, 0.5647512851104622],
                        [-3.475691078948082e-05, -2.556854941686433e-06, 0.35947258525400694],
                        [2.195590703160342e-05, -2.3103969855119255e-05, 0.11841425174599007],
                        [19.049644571924924e-06, -2.2489940539324337e-06, 0.6569198047733425],
                        [-3.209692195181183e-06, -2.885903407728193e-06, 0.3437838711835959],
                        [3.106069213627105e-05, -1.9967158454913738e-05, 0.45128984619758494],
                        [4.6047742666062404e-06, 5.874469697778904e-05, 0.671675637294559],
                        [-2.954057869579832e-06, -4.1005069370360806e-05, 0.3858047573349588],
                        [-5.43119013493178e-06, -4.6025618105172806e-05, 0.3705850310592902]]

    # k = [[1.178956203443989e-05, 4.059362224365455e-06, 0.4518755534305855],
    #      [-2.271810302613115e-05, 3.786735616103811e-05, 0.26239433992585426],
    #      [-2.780283293819153e-05, 1.4505684959253678e-05, 0.8432095919877094],
    #      [-6.095482930554621e-06, -1.92134835733238e-05, 0.6519152919184402],
    #      [8.496095888274791e-06, 1.0077434424164404e-05, 0.12849043981522945],
    #      [-3.77811841379065e-06, 5.517291154461439e-06, 0.9050594493584998],
    #      [2.7011455515135813e-05, -5.678518859535065e-06, 0.5647512851104622],
    #      [-5.475691078948082e-05, 7.556854941686433e-06, 0.35947258525400694],
    #      [-5.195590703160342e-05, -2.3103969855119255e-05, 0.11841425174599007],
    #      [-9.049644571924924e-06, -2.2489940539324337e-06, 0.8569198047733425],
    #      [6.209692195181183e-06, -8.885903407728193e-06, 0.3437838711835959],
    #      [-3.106069213627105e-05, 1.9967158454913738e-05, 0.45128984619758494],
    #      [4.6047742666062404e-05, 5.874469697778904e-05, 0.671675637294559],
    #      [-9.954057869579832e-06, -4.1005069370360806e-05, 0.3858047573349588],
    #      [-5.43119013493178e-05, 4.6025618105172806e-05, 0.3705850310592902]]

    route_data13, theory_route_data13, solve_result13, precision_target13 = motion.motionDataMoc.generateData13Line(
        data_len, sample_len, precision, axis_range=axis_range, pre_magnification=magnification,
        error_param=test_error_param, L=L)
    data_x, data_y = motion.motionDataMoc.generateDataAll(data_len, route_len, sample_len, precision, axis_range,
                                                          magnification, error_param=test_error_param, L=L)
    test_x, test_y, test_theory_route = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision,
                                                                             route=test_route,
                                                                             axis_range=axis_range,
                                                                             pre_magnification=magnification,
                                                                             error_param=test_error_param, L=L)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # 画出图形
    # plot_cnn_result(data_x.reshape(-1, data_x.shape[-1]), data_y.reshape(-1, data_y.shape[-1]), None, test_x,
    #                 test_y, solve_result13, label, axis_range, magnification, dir_path="image13Line")

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
    start_time = time.time()
    tensor_sample_x = torch.tensor(sample_x.tolist(), dtype=torch.double, device=device).unsqueeze(0)
    precision_grid = model.model(tensor_sample_x, mask).detach().cpu().numpy()
    elapsed_time = time.time() - start_time
    print(f"Transformer法运行时间: {elapsed_time:.6f} 秒")
    precision_grid = precision_grid.reshape(-1, precision_grid.shape[-1])
    test_x_new, test_pred_new, _t = motion.motionDataMoc.generate_pred_by_grid(precision_grid, data_len, sample_len,
                                                                               route=test_route,
                                                                               axis_range=axis_range,
                                                                               magnification=magnification)
    pred_by_comp_all = []

    route_comp = [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]
    for i in range(len(route_comp)):
        comp_route_x, comp_pred, comp_theory_route = motion.motionDataMoc.generate_pred_by_grid(precision_grid,
                                                                                                data_len, sample_len,
                                                                                                route=route_comp[i],
                                                                                                axis_range=axis_range,
                                                                                                magnification=magnification)
        comp = motion.motionDataMoc.generate_composition_value_by_grid(np.array(comp_pred), np.array(comp_route_x),
                                                                       route_comp[i],
                                                                       data_len,
                                                                       axis_range, magnification)
        pred_by_comp_all.append(comp)
    pred_by_comp_all = np.array(pred_by_comp_all)
    pred_comp = motion.motionDataMoc.generate_composition_value_by_all_comp(test_theory_route, pred_by_comp_all,
                                                                            data_len, axis_range, magnification)
    # 转为numpy对象
    train_pred = np.array(train_pred)
    test_pred = np.array(test_pred_new)
    avg = np.average(np.abs(test_pred - test_y))
    # 画出图形
    plot_cnn_result(data_x.reshape(-1, data_x.shape[-1]), data_y.reshape(-1, data_y.shape[-1]), train_pred, test_x,
                    test_y, test_pred, label, axis_range, magnification, L=L, pre_com_val=pred_comp,
                    theory_route=test_theory_route)
