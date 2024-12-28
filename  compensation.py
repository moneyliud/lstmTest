import torch
import motion.motionDataMoc
from cnn.PrecisionNN import PrecisionNN
import numpy as np
from plot.resultPlot import *


def normalize_data(data):
    return (data - data.min(0)) / (data.max(0) - data.min(0))


def generate_grid_points(grid_num, start_pos, range_size, grid_sample_num=None, sample_flag=False):
    grid_center_points = []
    grid_sample_points = []
    interval = range_size / grid_num
    interval_2 = interval / 2
    for x in range(grid_num):
        x_start = interval * x + start_pos[0]
        for y in range(grid_num):
            y_start = interval * y + start_pos[1]
            for z in range(grid_num):
                z_start = interval * z + start_pos[2]
                point = [x_start + interval_2, y_start + interval_2, z_start + interval_2]
                if sample_flag:
                    sample_points, _ = generate_grid_points(grid_sample_num, [x_start, y_start, z_start], interval)
                    grid_sample_points.append(sample_points)
                grid_center_points.append(point)
    return grid_center_points, grid_sample_points


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sample_len = 1
    # 各轴定位精度
    loc_pre = [0.12, 0.13, 0.11, 0, 0]
    # 直线度
    straightness = [[0.000005, 0.0000035], [0.0000024, 0.0000033], [0.0000022, 0.0000041]]
    # 角度偏差
    angleError = [[0.000003, 0.000002, 0.000002], [0.000004, 0.000002, 0.000002], [0.0000024, 0.000005, 0.0000031]]
    precision = [loc_pre, straightness, angleError]
    # 精度放大倍率，行程缩小倍率
    magnification = [1, 100000, 100000, 10000, 0.0001]
    # 各轴运动行程
    axis_range = [2000., 2000., 2000., 180., 180.]
    model_path = "ECNN_model.pt"
    weight_path = "ECNN_params.pt"
    model = torch.load(model_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    grid_num = 10
    grid_sample_num = 4
    all_data_x = []
    all_data_y = []

    grid_center_points, grid_sample_points = generate_grid_points(grid_num, [0, 0, 0], 1, grid_sample_num, True)

    grid_center_points = np.array(grid_center_points).reshape(-1, 3)
    grid_sample_points = np.array(grid_sample_points).reshape(-1, 3)

    center_x, center_y = motion.motionDataMoc.generateDataCNN(-1, -1, precision, points=grid_center_points,
                                                              axis_range=axis_range, pre_magnification=magnification)
    sample_center_x, sample_center_y = motion.motionDataMoc.generateDataCNN(-1, -1, precision,
                                                                            points=grid_center_points,
                                                                            axis_range=axis_range,
                                                                            pre_magnification=magnification)
    # print(center_x)
    print(sample_center_y)
    # tensor_test_x = torch.tensor(test_x, dtype=torch.double, device=device).unsqueeze(1).unsqueeze(1)
    # for i in range(tensor_train_x.shape[0]):
    #     # 预测训练数据
    #     train_pred.append(model.model(tensor_train_x[i]).detach().cpu().numpy())
    # print(center_y)
    # print(grid_sample_points)
    # 生成各个网格运动数据
    # for i in range(len(grid_center_points)):
    #     tmp_x, tmp_y = motion.motionDataMoc.generateDataCNN(200, sample_len, precision, points=grid_center_points,
    #                                                         axis_range=axis_range, pre_magnification=magnification)
    #     all_data_x.append(tmp_x)
    #     all_data_y.append(tmp_y)
