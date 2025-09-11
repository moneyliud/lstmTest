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
    straightness = [[0.000002, -0.000001], [-0.0000012, 0.0000014], [-0.0000008, 0.000001]]
    # 角度偏差
    angleError = [[0.000018, -0.00002, 0.000008], [0.000028, -0.000012, 0.00002], [0.000011, 0.000017, 0.000024]]
    # angleError = [[0.03, 0.02, 0.02], [0.04, 0.02, 0.02], [0.024, 0.05, 0.031]]
    # angleError = [[0., -0., 0.], [0.00004, -0.00002, 0.00002], [-0.0, 0.0, 0]]
    # 垂直度偏差
    # verticalPre = [0.00001, 0.00002, 0.00001, 0., 0., 0., 0.]
    # verticalPre = [0.0, 0.0, 0.0, 0., 0., 0., 0.]
    precision = [loc_pre, straightness, angleError]
    # 精度放大倍率，行程缩小倍率
    magnification = [5, 30000, 10000, 10000, 0.0001]
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
    route = [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]],
             [[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]]
    # route = [[[0, 0, 0], [1, 1, 1]],
    #          [[0, 0, 0], [1, 0, 1]],
    #          [[0, 0, 0], [1, 1, 0]],
    #          [[0, 0, 0], [0, 1, 1]]]
    # route = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1], [1, 1, 1]]
    data_len = 40
    # 多少组四线路径
    route_len = 30
    # 最大迭代次数
    max_epochs = 300
    # 每批数据数量
    batch_size = 5
    load_best = False
    # 学习率
    learning_rate = 0.0002
    test_route = [[0, 0, 0], [0.5, 1, 1]]
    test_route1 = [[0, 0, 0], [1, 0.5, 1]]
    test_route2 = [[0, 0, 0], [1, 1, 0.5]]
    test_route3 = [[0, 0, 0], [0.5, 0.5, 1]]
    # 生成一个路径的测试数据
    test_error_param = [[2.192216764043185, 0.2617623586880524, 0.6588709747758215],
                        [-1.4290295470882306, -4.547730223217397, 0.4450699781137264],
                        [3.9988284165462975, 1.695433768922194, 0.4],
                        # loc2
                        # [1.178956203443989e-05, 2.059362224365455e-06, 0.4518755534305855],
                        # [-2.271810302613115e-05, -3.786735616103811e-05, 0.26239433992585426],
                        # [-2.780283293819153e-05, -1.4505684959253678e-05, 0.8432095919877094],
                        # [4.095482930554621e-05, -1.92134835733238e-05, 0.6519152919184402],
                        # [8.496095888274791e-06, 1.0077434424164404e-05, 0.12849043981522945],
                        # [-3.77811841379065e-06, -2.517291154461439e-06, 0.4050594493584998],
                        # loc4
                        # 4
                        [[0,
                          0.06920027691827108,
                          0.12593402978162796,
                          0.19058873554112638,
                          0.25571351845297746,
                          0.3294742130675394,
                          0.4040066896925347,
                          0.47241192421923817,
                          0.5310102556358439,
                          0.5992703457647374,
                          0.6777174166996236,
                          0.7274064468547966,
                          0.8075348299559176,
                          0.869103254696862,
                          1],
                         [1.966019014703635e-06,
                          9.904999675058871e-07,
                          -6.055020270724317e-08,
                          -9.550511891217894e-07,
                          -1.5636020175947513e-06,
                          1.168450093301202e-06,
                          9.630570914805358e-07,
                          1.9167325826478667e-06,
                          -9.842233164032778e-07,
                          1.3395893596863528e-06,
                          8.656273791850421e-07,
                          1.6452152037665688e-06,
                          -6.33095285831843e-07,
                          -1.177997780906235e-06,
                          1.1572482226266103e-06]],
                        [[0,
                          0.09743452808908938,
                          0.1781168010224367,
                          0.2750819852078682,
                          0.36781210833660033,
                          0.44743150745600635,
                          0.5425322037871022,
                          0.6467103841246276,
                          0.7154028769714195,
                          0.8250740266757719,
                          1],
                         [1.8267435052280026e-06,
                          -4.4335443459530084e-07,
                          -1.4185122446743e-06,
                          -1.9366891373924997e-06,
                          1.0568960637031591e-06,
                          -6.759604060073956e-07,
                          6.010020553038191e-07,
                          -1.0855012621598749e-06,
                          8.818415153984986e-07,
                          5.41856295606439e-07,
                          -9.434740742606449e-07]],
                        [[0,
                          0.07915129670234994,
                          0.13886412544212595,
                          0.20582705797674175,
                          0.27681015392136954,
                          0.36366287556575816,
                          0.4230101572607913,
                          0.5102848948931341,
                          0.5794008196832591,
                          0.643782085525472,
                          0.7188581882046261,
                          0.797607240446644,
                          0.8673773725134727,
                          1],
                         [7.37281100297839e-07,
                          5.090417460146818e-07,
                          8.247200588764699e-07,
                          2.6109183722380205e-08,
                          -1.9037024583553865e-06,
                          -8.549210432625686e-07,
                          2.3624448742856402e-07,
                          -2.345345923047751e-07,
                          -9.78860533064657e-07,
                          1.9275203474889146e-06,
                          -4.6278965613967573e-07,
                          -0.7623730227612463e-06,
                          0.584089308773427e-06,
                          -7.164026076231237e-07]],
                        [[0,
                          0.08292829721027413,
                          0.1689237547273678,
                          0.2517619819280472,
                          0.33550608587739505,
                          0.40472345704029705,
                          0.5095556501245568,
                          0.5941258312176084,
                          0.6626255506726386,
                          0.7544281945974903,
                          0.8398595779684627,
                          1],
                         [5.718067513064415e-07,
                          -5.249937129899824e-07,
                          2.1725712786192213e-07,
                          2.2336116335126e-07,
                          3.5212268875540147e-07,
                          -2.2411853241531698e-07,
                          -1.418874593635734e-06,
                          -1.8537616607698939e-06,
                          1.2378533997459532e-06,
                          1.558670085119503e-06,
                          -4.325562669105967e-07,
                          -3.6094042590229275e-07]],
                        [[0,
                          0.08086343260955982,
                          0.131497537482842,
                          0.2045438900247505,
                          0.28753271729518054,
                          0.34629317687636374,
                          0.4299660427117168,
                          0.4983258946063051,
                          0.5737194840578949,
                          0.6365208107883165,
                          0.7251964590500438,
                          0.7834481886719074,
                          0.8459838541589181,
                          1],
                         [5.051158623012873e-07,
                          -9.921288623552468e-07,
                          1.3101934627712535e-06,
                          1.667165103855427e-07,
                          -6.947662565330849e-07,
                          -3.9012045084488947e-07,
                          7.452851980314721e-07,
                          7.345712895825889e-07,
                          8.944222299342247e-07,
                          -3.020637054019544e-08,
                          5.019569033424186e-07,
                          6.123104422600794e-07,
                          1.1868304377164056e-06,
                          5.394805056284336e-07]],
                        [[0,
                          0.07090590287674911,
                          0.12409561385170903,
                          0.20810746107272746,
                          0.256618408847367,
                          0.32550617591417996,
                          0.40109198802632795,
                          0.4646173023280431,
                          0.5429295952000968,
                          0.6037543803679617,
                          0.6693577948431308,
                          0.733559811382477,
                          0.7965725768705909,
                          0.8636810604989286,
                          1],
                         [-7.510073728108402e-08,
                          8.431131714946876e-08,
                          -2.5160815279573344e-07,
                          -1.2618113441028705e-06,
                          -1.0298576345748994e-07,
                          1.5087436169968704e-06,
                          8.42165047882558e-07,
                          -2.4337460060623254e-07,
                          -1.3568641861546563e-06,
                          -1.992750768345899e-06,
                          1.7205478262838483e-06,
                          -1.0512571134028706e-06,
                          8.619079506681116e-07,
                          1.6835344273897975e-07,
                          1.257607709043438e-06]],
                        # 10
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
    _a1, test_pred_new1, _t1 = motion.motionDataMoc.generate_pred_by_grid(precision_grid, data_len, sample_len,
                                                                          route=test_route1,
                                                                          axis_range=axis_range,
                                                                          magnification=magnification)
    _a2, test_pred_new2, _t2 = motion.motionDataMoc.generate_pred_by_grid(precision_grid, data_len, sample_len,
                                                                          route=test_route2,
                                                                          axis_range=axis_range,
                                                                          magnification=magnification)
    _a3, test_pred_new3, _t3 = motion.motionDataMoc.generate_pred_by_grid(precision_grid, data_len, sample_len,
                                                                          route=test_route3,
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
    test_pred_all = [np.array(test_pred_new), np.array(test_pred_new1), np.array(test_pred_new2),
                     np.array(test_pred_new3)]
