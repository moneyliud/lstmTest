from motion.motionCalculator import MotionCalculator
import numpy as np
import random
import torch


# 增加随机误差
def add_error(a, rate=0.008):
    return a + (random.random() * rate * 2 - rate)


# 线性误差模拟函数
def locFunc(value, b, a=0.5):
    return add_error(value * a / 600 + b)


# 线性误差模拟函数1
def locFunc1(value, b, a1=300, a2=-400, thres=130, rate=0.002):
    b2 = thres * b / a1 + b - thres * b / a2
    if value < thres:
        return add_error(value * b / a1 + b, rate)
    return add_error(value * b / a2 + b2, rate)


# 生成LSTM所需训练数据
def generateDataLSTM(data_len, sample_len, presicion, next_step=0, route=None):
    data_x, data_y = generateData(data_len, sample_len, presicion, route)
    if type == "LSTM":
        for i in range(data_len):
            data_x.append(data_x[i:i + sample_len])
            data_y.append([data_y[i + sample_len + next_step]])

        data_x = np.array(data_x).astype('float32')
        data_y = np.array(data_y).astype('float32')
        tmp_data_x = np.array(data_x).astype('float32')
        return data_x, data_y, tmp_data_x[:data_len]


# 生成CNN所需训练数据
def generateDataCNN(data_len, sample_len, presicion, next_step=0, route=None, axis_range=None, pre_magnification=None):
    data_x, data_y = generateData(data_len, sample_len, presicion, next_step, route, axis_range, pre_magnification)
    return data_x, data_y


# 生成训练数据集，参数：[数据长度，lstm采样长度，设备精度数组，预测的下一组数据跳过的步数，机床运行路径]
def generateData(data_len, sample_len, presicion, next_step=0, route=None, axis_range=None, magnification=None):
    # 机床运动计算模型
    calculator = MotionCalculator()
    axisRange = axis_range
    calculator.setAxisRange(axisRange)
    # 定位误差
    loc_pre = presicion[0]
    # 直线度
    straightness = presicion[1]
    # 角度偏差
    angle_error = presicion[2]
    # 垂直度偏差
    vertical_pre = presicion[3]
    tmp_data_x = []
    tmp_data1_y = []
    interval = int(2000 / (data_len + sample_len + next_step))

    for i in range(data_len + sample_len + next_step):
        # a = testFunc((i - 1) * math.pi / step)
        in_loc = [locFunc1(i, loc_pre[0], a1=300, a2=500, thres=50, rate=0.0005),
                  locFunc1(i, loc_pre[1], a1=400, a2=-500, thres=60, rate=0.0005),
                  locFunc1(i, loc_pre[2], a1=500, a2=-300, thres=70, rate=0.0005), 0, 0]
        in_straightness = [[locFunc1(i, straightness[0][0], a1=300, a2=-340, thres=30, rate=0.0000002),
                            locFunc1(i, straightness[0][1], a1=400, a2=500, thres=80, rate=0.0000002)],
                           [locFunc1(i, straightness[1][0], a1=600, a2=-400, thres=50, rate=0.0000002),
                            locFunc1(i, straightness[1][1], a1=340, a2=-670, thres=60, rate=0.0000002)],
                           [locFunc1(i, straightness[2][0], a1=890, a2=300, thres=50, rate=0.0000002),
                            locFunc1(i, straightness[2][1], a1=360, a2=500, thres=40, rate=0.0000002)]]
        in_angle_error = [[locFunc1(i, angle_error[0][0], a1=300, a2=-340, thres=60, rate=0.0000002),
                           locFunc1(i, angle_error[0][1], a1=400, a2=500, thres=70, rate=0.0000002)],
                          [locFunc1(i, angle_error[1][0], a1=200, a2=-200, thres=120, rate=0.0000002),
                           locFunc1(i, angle_error[1][1], a1=640, a2=-370, thres=33, rate=0.0000002)],
                          [locFunc1(i, angle_error[2][0], a1=790, a2=620, thres=55, rate=0.0000002),
                           locFunc1(i, angle_error[2][1], a1=660, a2=340, thres=66, rate=0.0000002)]]
        in_vertical_pre = [locFunc1(i, vertical_pre[0], a1=370, a2=300, thres=50, rate=0.000002),
                           locFunc1(i, vertical_pre[1], a1=450, a2=300, thres=66, rate=0.000002),
                           locFunc1(i, vertical_pre[2], a1=350, a2=500, thres=44, rate=0.000002), 0, 0, 0, 0]
        # input = [locFunc(i, 0.05), 0.2, 0.3, 0.1, 0.4]
        # 设置定位精度
        calculator.setPrecision(pLoc=in_loc, straightness=in_straightness, angleError=in_angle_error,
                                verticalPre=in_vertical_pre)
        x = i * interval
        y = i * interval
        z = i * interval
        if route is not None:
            x *= route[0]
            y *= route[1]
            z *= route[2]
        error, m1, m2, m3 = calculator.calculate(x, y, z, 0., 0.)
        # tmp_data_x.append([error, m1, m2, m3, x, y, z])
        tmp_data_x.append([error, m1, m2, m3, x * magnification[4], y * magnification[4], z * magnification[4]])
        target = np.concatenate(
            (np.array(in_loc)[0:3] * magnification[0],
             np.array(in_straightness).reshape(-1) * magnification[1],
             np.array(in_angle_error).reshape(-1) * magnification[2],
             np.array(in_vertical_pre)[0:3] * magnification[3]), 0, None)
        tmp_data1_y.append(target.tolist())
        # tmp_data1_y.append([input[0]])
    return tmp_data_x, tmp_data1_y
