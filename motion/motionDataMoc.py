import math

from motion.motionCalculator import MotionCalculator
import numpy as np
import random
import torch
from motion.Method13LineSolver import Method13LineSolver
import motion


# 增加随机误差
def add_error(a, rate=0.008):
    return a + (random.random() * rate * 2 - rate)


# 线性误差模拟函数
def locFunc(value, b, a=0.5):
    return add_error(value * a / 600 + b)


# 线性误差模拟函数1
def locFunc1(value, b, a1=3.0, a2=-4.0, thres=0.5, rate=0.002):
    b2 = thres * b * a1 + b - thres * b * a2
    if value < thres:
        return add_error(value * b * a1 + b, rate)
    return add_error(value * b * a2 + b2, rate)


def array_to_precession_input(pre_array, magnification):
    return np.concatenate((pre_array[0:3], [0, 0]), axis=0) / magnification[0], \
           pre_array[3:9].reshape(-1, 2) / magnification[1], \
           pre_array[9:18].reshape(-1, 3) / magnification[2]


# 生成LSTM所需训练数据
def generateDataLSTM(data_len, sample_len, precision, next_step=0, route=None):
    data_x, data_y, theory_route = generateData(data_len, sample_len, precision, route)
    if type == "LSTM":
        for i in range(data_len):
            data_x.append(data_x[i:i + sample_len])
            data_y.append([data_y[i + sample_len + next_step]])

        data_x = np.array(data_x).astype('float32')
        data_y = np.array(data_y).astype('float32')
        tmp_data_x = np.array(data_x).astype('float32')
        return data_x, data_y, tmp_data_x[:data_len]


# 生成13线法的辨识结果数据
def generateData13Line(data_len, sample_len, precision, next_step=0, axis_range=None,
                       pre_magnification=None,
                       points=None, error_param=None):
    route = [
        # X轴第一线，L1
        [[0, 0, 0], [1, 0, 0]],
        # X轴第二线，L2
        [[0, 1, 0], [1, 1, 0]],
        # X轴第三线，L3
        [[0, 0, 1], [1, 0, 1]],
        # Y轴第一线，L4
        [[0, 0, 0], [0, 1, 0]],
        # Y轴第二线，L5
        [[1, 0, 0], [1, 1, 0]],
        # Y轴第三线，L6
        [[0, 0, 1], [0, 1, 1]],
        # Z轴第一线，L7
        [[0, 0, 0], [0, 0, 1]],
        # Z轴第二线，L8
        [[1, 0, 0], [1, 0, 1]],
        # Z轴第三线，L9
        [[0, 1, 0], [0, 1, 1]],
        # XY面对角线，L10
        [[0, 0, 0], [1, 1, 0]],
        # XZ面对角线，L11
        [[0, 0, 0], [1, 0, 1]],
        # ZY面对角线，L12
        [[0, 0, 0], [0, 1, 1]],
        # XYZ体对角线，L13
        [[0, 0, 0], [1, 1, 1]]]
    if pre_magnification is not None:
        pre_magnification[4] = 1
    route_data = None
    theory_route_data = None
    precision_target = []
    solve_result = []
    for i in range(len(route)):
        data_x, data_y, theory_route = generateData(data_len, sample_len, precision, next_step, route[i], axis_range,
                                                    pre_magnification,
                                                    points, error_param)
        data_x = np.array(data_x)
        if route_data is None:
            route_data = np.expand_dims(data_x[:, 1:4], axis=1)
            theory_route_data = np.expand_dims(theory_route, axis=1)
        else:
            route_data = np.concatenate((route_data, np.expand_dims(data_x[:, 1:4], axis=1)), axis=1)
            theory_route_data = np.concatenate((theory_route_data, np.expand_dims(theory_route, axis=1)), axis=1)
        data_y = np.array(data_y)
        if i == len(route) - 1:
            precision_target = data_y

    for i in range(len(route_data)):
        tmp = Method13LineSolver.solve(route_data[i], theory_route_data[i], axis_range)
        tmp[0:3] = [x * pre_magnification[0] for x in tmp[0:3]]
        tmp[3:9] = [x * pre_magnification[1] for x in tmp[3:9]]
        tmp[9:18] = [x * pre_magnification[2] for x in tmp[9:18]]
        solve_result.append(tmp)
    precision_target = np.array(precision_target)
    solve_result = np.array(solve_result)
    return route_data, theory_route_data, solve_result, precision_target


# 生成CNN所需训练数据
def generateDataCNN(data_len, sample_len, precision, next_step=0, route=None, axis_range=None, pre_magnification=None,
                    points=None, error_param=None):
    data_x, data_y, theory_route = generateData(data_len, sample_len, precision, next_step, route, axis_range,
                                                pre_magnification,
                                                points, error_param)
    return data_x, data_y, theory_route


def generateDataAll(data_len, route_len, sample_len, precision, axis_range, magnification, error_param=None):
    # 运动路线，为直线运动的终点坐标，取0-1，表示各轴的行程范围
    route = [[[0, 0, 0], [1, 1, 1]],
             [[0, 0, 0], [1, 0, 1]],
             [[0, 0, 0], [1, 1, 0]],
             [[0, 0, 0], [0, 1, 1]]]
    all_data_x = []
    all_data_y = []
    # 生成各轴运动数据
    for k in range(route_len):
        # 18项误差的随机值
        # loc_pre_random = []
        # straightness_random = []
        # angle_error_random = []
        # precision_random = [loc_pre_random, straightness_random, angle_error_random]
        precision_random = precision
        # 18项误差线性变化随机值
        if error_param is None:
            error_params_random = [[(1 if random.random() - 0.5 > 0 else -1) * random.random() * 6,
                                    (1 if random.random() - 0.5 > 0 else -1) * random.random() * 6, random.random()]
                                   for n in range(18)]
            # error_params_random = [[random.random() * 6,
            #                         random.random() * 6, random.random()]
            #                        for n in range(18)]
        else:
            error_params_random = error_param
        target_y = None
        tmp_data_x = []
        for i in range(len(route)):
            tmp_x, tmp_y, theory_route = motion.motionDataMoc.generateDataCNN(data_len, sample_len, precision_random,
                                                                              route=route[i],
                                                                              axis_range=axis_range,
                                                                              pre_magnification=magnification,
                                                                              error_param=error_params_random)
            tmp_data_x.append(tmp_x)
            if i == 0:
                target_y = tmp_y
        tmp_data_x = np.array(tmp_data_x)
        tmp_data_x = np.transpose(tmp_data_x, (1, 0, 2))
        tmp_data_x = tmp_data_x.reshape(-1, len(route) * tmp_data_x.shape[-1])
        all_data_x.append(tmp_data_x.tolist())
        all_data_y.append(target_y)
    # 各轴运动路径、末端误差数据，整理为一个输入
    data_x = np.array(all_data_x)
    # 各轴误差数据作为标签数据，整理为一个数组
    data_y = np.array(all_data_y)
    return data_x, data_y


# 生成训练数据集，参数：[数据长度，lstm采样长度，设备精度数组，预测的下一组数据跳过的步数，机床运行路径]
def generateData(data_len, sample_len, precision, next_step=0, route=None, axis_range=None, magnification=None,
                 points=None, error_param=None):
    # 机床运动计算模型
    calculator = MotionCalculator()
    calculator.setAxisRange(axis_range)
    # 定位误差
    loc_pre = precision[0]
    # 直线度
    straightness = precision[1]
    # 角度偏差
    angle_error = precision[2]
    # 垂直度偏差
    # vertical_pre = precision[3]
    tmp_data_x = []
    theory_route = []
    tmp_data1_y = []
    interval = []
    for i in range(len(axis_range)):
        interval.append(axis_range[i] / (data_len + sample_len + next_step))
    total_len = data_len + sample_len + next_step
    if points is not None:
        total_len = len(points)
    x_s, y_s, z_s = route[0][0], route[0][1], route[0][2]
    dir_x, dir_y, dir_z = route[1][0] - route[0][0], route[1][1] - route[0][1], route[1][2] - route[0][2]
    for i in range(total_len):
        x, y, z = 0, 0, 0
        if points is not None:
            x = points[i][0] * axis_range[0]
            y = points[i][1] * axis_range[1]
            z = points[i][2] * axis_range[2]
        else:
            x = i * interval[0]
            y = i * interval[1]
            z = i * interval[2]
            if route is not None:
                x = x_s * axis_range[0] + x * dir_x
                y = y_s * axis_range[1] + y * dir_y
                z = z_s * axis_range[2] + z * dir_z

        x_rate = x / axis_range[0]
        y_rate = y / axis_range[1]
        z_rate = z / axis_range[2]
        # a = testFunc((i - 1) * math.pi / step)
        in_loc = [0] * 5
        pos = [x_rate, y_rate, z_rate]
        # 3项各轴的定位误差
        for k in range(3):
            in_loc[k] = locFunc1(pos[k], loc_pre[k], a1=error_param[k][0], a2=error_param[k][1],
                                 thres=error_param[k][2], rate=0.0005)
        in_straightness = [[0, 0] for i in range(3)]
        # 6项各轴的直线度误差
        for k in range(3):
            start = k * 2 + 3
            in_straightness[k] = [
                locFunc1(pos[k], straightness[k][0], a1=error_param[start][0], a2=error_param[start][1],
                         thres=error_param[start][2], rate=0.0000002),
                locFunc1(pos[k], straightness[k][1], a1=error_param[start + 1][0], a2=error_param[start + 1][1],
                         thres=error_param[start + 1][2], rate=0.0000002)]
        index = 3 * 2 + 3
        in_angle_error = [[0, 0, 0] for i in range(3)]
        # 9项各轴的角度误差
        for k in range(3):
            for t in range(3):
                in_angle_error[k][t] = locFunc1(x_rate, angle_error[k][t], a1=error_param[index][0],
                                                a2=error_param[index][1], thres=error_param[index][2], rate=0.0000001)
                index = index + 1
        # in_vertical_pre = [locFunc1(i, vertical_pre[0], a1=370, a2=300, thres=50, rate=0.000002),
        #                    locFunc1(i, vertical_pre[1], a1=450, a2=300, thres=66, rate=0.000002),
        #                    locFunc1(i, vertical_pre[2], a1=350, a2=500, thres=44, rate=0.000002), 0, 0, 0, 0]
        # input = [locFunc(i, 0.05), 0.2, 0.3, 0.1, 0.4]
        # 设置定位精度
        calculator.setPrecision(pLoc=in_loc, straightness=in_straightness, angleError=in_angle_error)
        calculator.setBiasACY(0)
        calculator.setL(0)
        error, m1, m2, m3, error_pjct, x_act, y_act, z_act = calculator.calculate(x, y, z, 0., 0.)
        # tmp_data_x.append([error, m1, m2, m3, x, y, z])
        point_dis = math.sqrt(x ** 2 + y ** 2 + z ** 2) * magnification[4]
        # tmp_data_x.append(
        #     [error, m1, m2, m3, x * magnification[4], y * magnification[4], z * magnification[4], point_dis])
        theory_route.append([x, y, z])
        tmp_data_x.append(
            [error_pjct, x_act * magnification[4], y_act * magnification[4], z_act * magnification[4]])
        target = np.concatenate(
            (np.array(in_loc)[0:3] * magnification[0],
             np.array(in_straightness).reshape(-1) * magnification[1],
             np.array(in_angle_error).reshape(-1) * magnification[2]), 0, None)
        tmp_data1_y.append(target.tolist())
        # tmp_data1_y.append([input[0]])
    return tmp_data_x, tmp_data1_y, theory_route


def generate_pred_by_grid(precision_grid, data_len, sample_len, next_step=0, route=None, axis_range=None,
                          magnification=None):
    interval = []
    tmp_data_x = []
    tmp_data_y = []

    for i in range(len(axis_range)):
        interval.append(axis_range[i] / (data_len + sample_len + next_step))
    total_len = data_len + sample_len + next_step
    x_s, y_s, z_s = route[0][0], route[0][1], route[0][2]
    dir_x, dir_y, dir_z = route[1][0] - route[0][0], route[1][1] - route[0][1], route[1][2] - route[0][2]
    for i in range(total_len):
        x, y, z, x_idx, y_idx, z_idx = 0, 0, 0, 0, 0, 0
        x = i * interval[0]
        y = i * interval[1]
        z = i * interval[2]
        if route is not None:
            x = x_s * axis_range[0] + x * dir_x
            y = y_s * axis_range[1] + y * dir_y
            z = z_s * axis_range[2] + z * dir_z
            x_idx = int(x / interval[0])
            y_idx = int(y / interval[1])
            z_idx = int(z / interval[2])
        tmp_data_x.append([0, 0, 0, 0, x * magnification[4], y * magnification[4], z * magnification[4]])
        # "x", "y", "z", "S_XXY", "S_XXZ", "S_YYZ", "S_YYX", "S_ZZY", "S_ZZX",
        # "A_XA", "A_XB", "A_XC", "A_YA","A_YB", "A_YC", "A_ZA", "A_ZB", "A_ZC"
        loc = [precision_grid[x_idx][0], precision_grid[y_idx][1], precision_grid[z_idx][2]]
        straightness = [precision_grid[x_idx][3], precision_grid[x_idx][4],
                        precision_grid[y_idx][5], precision_grid[y_idx][6],
                        precision_grid[z_idx][7], precision_grid[z_idx][8]]
        angle_error = [precision_grid[x_idx][9], precision_grid[x_idx][10], precision_grid[x_idx][11],
                       precision_grid[y_idx][12], precision_grid[y_idx][13], precision_grid[y_idx][14],
                       precision_grid[z_idx][15], precision_grid[z_idx][16], precision_grid[z_idx][17]]
        target = np.concatenate(
            (np.array(loc),
             np.array(straightness).reshape(-1),
             np.array(angle_error).reshape(-1)), 0, None)
        tmp_data_y.append(target.tolist())
    return tmp_data_x, tmp_data_y
