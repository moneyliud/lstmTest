import math

import matplotlib.pyplot as plt
import numpy as np

from motion.motionCalculator import MotionCalculator
from motion.motionDataMoc import array_to_precession_input
import os.path


def draw_one_graph(y_data, pred_y_data, y_label, title, dir_path):
    plt.figure(figsize=(9, 6))
    color = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#000000', '#880000', '#008800',
             '#000088']
    test_t = np.linspace(0, y_data.shape[0] - 1, y_data.shape[0])
    for i in range(y_data.shape[1]):
        test_y_i = y_data[:, i].reshape(-1)
        pred_y_for_test_i = pred_y_data[:, i].reshape(-1)
        plt.plot(test_t, test_y_i, label=y_label[i], color=color[i])
        plt.plot(test_t, pred_y_for_test_i, '--', label=y_label[i] + '_预测值', color=color[i])
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right', borderaxespad=0.)
    plt.xlabel('分段序号')
    plt.ylabel('误差值(mm)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{dir_path}/{title}.png")
    pass


def plot_cnn_result(train_x, train_y, pred_y_for_train, test_x, test_y, pred_y_for_test, y_label,
                    axis_range, magnification, dir_path="image", L=0, pre_com_val=None, theory_route=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.grid(True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用黑体
    draw_one_graph(test_y[:, 0:3] / magnification[0], pred_y_for_test[:, 0: 3] / magnification[0],
                   y_label[0:3], "定位误差(mm)", dir_path)
    draw_one_graph(test_y[:, 3:9] / magnification[1], pred_y_for_test[:, 3:9] / magnification[1],
                   y_label[3:9], "直线度(mm)", dir_path)
    draw_one_graph(test_y[:, 9:18] / magnification[2], pred_y_for_test[:, 9:18] / magnification[2],
                   y_label[9:18], "角度误差(mm)", dir_path)
    # draw_one_graph(test_y[:, 15:18], pred_y_for_test[:, 15:18], y_label[15:18], "vertical error")
    # plt.figure()
    color = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#000000', '#880000', '#008800',
             '#000088']
    #
    test_t = np.linspace(0, test_y.shape[0] - 1, test_y.shape[0])
    # for i in range(test_y.shape[1]):
    #     test_y_i = test_y[:, i].reshape(-1)
    #     pred_y_for_test_i = pred_y_for_test[:, i].reshape(-1)
    #     plt.plot(test_t, test_y_i, label=y_label[i], color=color[i])
    #     plt.plot(test_t, pred_y_for_test_i, '--', label=y_label[i] + '_test', color=color[i])
    #
    # plt.legend()
    # plt.xlabel('test_x')
    # plt.ylabel('text_y')

    # plt.figure()
    # train_t = np.linspace(0, train_y.shape[0] - 1, train_y.shape[0])
    # for i in range(train_y.shape[1]):
    #     train_y_i = train_y[:, i].reshape(-1)
    #     pred_y_for_train_i = pred_y_for_train[:, i].reshape(-1)
    #     plt.plot(train_t, train_y_i, label=y_label[i], color=color[i])
    #     plt.plot(train_t, pred_y_for_train_i, '--', label=y_label[i] + '_train', color=color[i])
    # plt.legend()
    # plt.xlabel('train_x')
    # plt.ylabel('train_y')

    # plt.figure()
    # for i in range(4):
    #     plt.plot(t, error_value[:, i], 'b', label='input')
    # plt.xlabel('t')
    # plt.ylabel('input')
    idx = 1

    if train_x.shape[1] > 4:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(train_x[:, 1], train_x[:, 2], train_x[:, 3])
        ax.plot(test_x[:, 1], test_x[:, 2], test_x[:, 3])
        ax.set_xlabel('X轴(mm)')
        ax.set_ylabel('Y轴(mm)')
        ax.set_zlabel('Z轴(mm)')
        plt.savefig(f"{dir_path}/result{str(idx)}.png")
        idx += 1
        # for i in range(error_value.shape[-1]):
        #     plt.figure()
        #     plt.plot(t, error_value[:, i], 'b', label='input' + str(i))
        #     plt.xlabel('t')
        #     plt.ylabel('input')

    # 画综合误差
    calculator = MotionCalculator()
    calculator.setBiasACY(0)
    calculator.setL(L)
    calculator.setAxisRange(axis_range)
    pre_len = 5
    theory_pre = np.zeros((pre_len, test_x.shape[0]))
    pred_pre = np.zeros((pre_len, test_x.shape[0]))
    error_percent = np.zeros((pre_len, test_x.shape[0]))
    error_abs = np.zeros((pre_len, test_x.shape[0]))
    # 补偿后的位置误差
    error_abs_comp = np.zeros((4, test_x.shape[0]))
    label = ["刀尖点误差(mm)", "X向误差(mm)", "Y向误差(mm)", "Z向误差(mm)", "运动方向投影误差(mm)"]
    label_percent = ["刀尖点误差预测准确率(%)", "X向误差预测准确率(%)", "Y向误差预测准确率(%)", "Z向误差预测准确率(%)",
                     "运动方向投影误差预测准确率(%)"]
    label_error_abs = ["刀尖点误差差值(mm)", "X向误差差值(mm)", "Y向误差差值(mm)", "Z向误差差值(mm)",
                       "运动方向投影误差差值(mm)"]
    label_error_abs_comp = ["补偿后刀尖点误差(mm)", "补偿后X向误差(mm)", "补偿后Y向误差(mm)", "补偿后Z向误差(mm)"]
    tmp_error18 = pred_y_for_test - test_y
    tmp_error_max = np.max(tmp_error18, 0)
    tmp_error_min = np.min(tmp_error18, 0)
    max_loc, max_straightness, max_angle = array_to_precession_input(tmp_error_max,
                                                                     magnification)
    min_loc, min_straightness, min_angle = array_to_precession_input(tmp_error_min,
                                                                     magnification)
    print(max_loc, max_straightness, max_angle)
    print(min_loc, min_straightness, min_angle)
    for i in range(test_x.shape[0]):
        pred_loc, pred_straightness, pred_angle = array_to_precession_input(pred_y_for_test[i],
                                                                            magnification)
        test_loc, test_straightness, test_angle = array_to_precession_input(test_y[i], magnification)
        calculator.setPrecision(pLoc=pred_loc, straightness=pred_straightness, angleError=pred_angle)
        pre = calculator.calculate(theory_route[i][0] / magnification[4], theory_route[i][1] / magnification[4],
                                   theory_route[i][2] / magnification[4], 0., 0.)
        for k in range(pre_len):
            pred_pre[k][i] = pre[k]
        calculator.setPrecision(pLoc=test_loc, straightness=test_straightness, angleError=test_angle)
        actual = calculator.calculate(theory_route[i][0] / magnification[4], theory_route[i][1] / magnification[4],
                                      theory_route[i][2] / magnification[4], 0., 0.)
        for k in range(pre_len):
            theory_pre[k][i] = actual[k]
            if math.fabs(actual[k]) < 0.01:
                error_percent[k][i] = 100
            else:
                error_percent[k][i] = 100 - math.fabs((pre[k] - actual[k]) / actual[k] * 100)
            error_abs[k][i] = (pre[k] - actual[k])
        if pre_com_val is not None:
            error_abs_comp[1][i] = -(actual[5] - pre_com_val[i][0] - theory_route[i][0])
            error_abs_comp[2][i] = -(actual[6] - pre_com_val[i][1] - theory_route[i][1])
            error_abs_comp[3][i] = -(actual[7] - pre_com_val[i][2] - theory_route[i][2])
            error_abs_comp[0][i] = math.sqrt(pow(error_abs_comp[1][i], 2) + pow(error_abs_comp[2][i], 2) + pow(
                error_abs_comp[3][i], 2))

    if pre_com_val is not None:
        for i in range(error_abs_comp.shape[0]):
            plt.figure()
            plt.plot(test_t, error_abs_comp[i], label=label_error_abs_comp[i], color="#FF0000")
            plt.xlabel('分段序号')
            plt.ylabel(label_error_abs_comp[i])
            plt.ylim(-0.1, 0.1)
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{dir_path}/result{str(idx)}.png")
            plt.close()
            idx += 1

    for i in range(pred_pre.shape[0]):
        tmp_error = theory_pre[i] - pred_pre[i]
        max_error = np.max(tmp_error)
        min_error = np.min(tmp_error)
        percent_e = np.max(np.abs(tmp_error) / theory_pre[i])
        print(label[i], max_error, min_error, percent_e)

        plt.figure(figsize=(9, 6))
        plt.plot(test_t, theory_pre[i], label="理论误差(mm)", color="#FF0000")
        plt.plot(test_t, pred_pre[i], '--', label='预测误差(mm)', color="#FF0000")
        plt.xlabel('分段序号')
        plt.ylabel(label[i])
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{dir_path}/result{str(idx)}.png")
        idx += 1

        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(test_t, error_abs[i], label=label_error_abs[i], color="#FF0000")
        ax1.set_xlabel('分段序号')
        ax1.set_ylabel(label_error_abs[i])
        ax1.set_ylim(-0.1, 0.1)
        ax1.set_yticks(np.arange(-0.1, 0.1, 0.02))
        ax1.legend()
        ax1.grid(True)
        # plt.figure()
        # plt.savefig(f"{dir_path}/result{str(idx)}.png")
        # idx += 1

        # plt.figure()
        ax2 = ax1.twinx()
        ax2.plot(test_t, error_percent[i], '--', label="误差预测准确率(%)", color="#FF0000")
        ax2.set_xlabel('分段序号')
        ax2.set_ylabel(label_percent[i])
        ax2.set_yticks(np.arange(10, 120, 10))
        ax2.legend(loc='upper left')
        ax2.grid(True)
        plt.savefig(f"{dir_path}/result{str(idx)}.png")
        idx += 1
        plt.close()
