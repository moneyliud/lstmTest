import matplotlib.pyplot as plt
import numpy as np

from motion.motionCalculator import MotionCalculator
from motion.motionDataMoc import array_to_precession_input
import os.path


def draw_one_graph(y_data, pred_y_data, y_label, title, dir_path):
    plt.figure()
    color = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#000000', '#880000', '#008800',
             '#000088']
    test_t = np.linspace(0, y_data.shape[0] - 1, y_data.shape[0])
    for i in range(y_data.shape[1]):
        test_y_i = y_data[:, i].reshape(-1)
        pred_y_for_test_i = pred_y_data[:, i].reshape(-1)
        plt.plot(test_t, test_y_i, label=y_label[i], color=color[i])
        plt.plot(test_t, pred_y_for_test_i, '--', label=y_label[i] + '_pred', color=color[i])
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.savefig(f"{dir_path}/{title}.png")
    pass


def plot_cnn_result(train_x, train_y, pred_y_for_train, test_x, test_y, pred_y_for_test, y_label,
                    axis_range, magnification, dir_path="image"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.grid(True)
    draw_one_graph(test_y[:, 0:3] / magnification[0], pred_y_for_test[:, 0: 3] / magnification[0],
                   y_label[0:3], "loc", dir_path)
    draw_one_graph(test_y[:, 3:9] / magnification[1], pred_y_for_test[:, 3:9] / magnification[1],
                   y_label[3:9], "straightness", dir_path)
    draw_one_graph(test_y[:, 9:18] / magnification[2], pred_y_for_test[:, 9:18] / magnification[2],
                   y_label[9:18], "angle error", dir_path)
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
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
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
    calculator.setL(0)
    calculator.setAxisRange(axis_range)
    pre_len = 5
    theory_pre = np.zeros((pre_len, test_x.shape[0]))
    pred_pre = np.zeros((pre_len, test_x.shape[0]))
    error_percent = np.zeros((pre_len, test_x.shape[0]))
    error_abs = np.zeros((pre_len, test_x.shape[0]))
    label = ["pos_error", "x_error", "y_error", "z_error", "project_error"]
    label_percent = ["pos_error_percent", "x_error_percent", "y_error_percent", "z_error_percent",
                     "project_error_percent"]
    label_error_abs = ["pos_error", "x_error", "y_error", "z_error",
                       "project_error"]
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
        pre = calculator.calculate(test_x[i][1] / magnification[4], test_x[i][2] / magnification[4],
                                   test_x[i][3] / magnification[4], 0., 0.)
        for k in range(pre_len):
            pred_pre[k][i] = pre[k]
        calculator.setPrecision(pLoc=test_loc, straightness=test_straightness, angleError=test_angle)
        actual = calculator.calculate(test_x[i][1] / magnification[4], test_x[i][2] / magnification[4],
                                      test_x[i][3] / magnification[4], 0., 0.)
        for k in range(pre_len):
            theory_pre[k][i] = actual[k]
            error_percent[k][i] = (pre[k] - actual[k]) / actual[k] * 100
            error_abs[k][i] = (pre[k] - actual[k])
    for i in range(pred_pre.shape[0]):
        tmp_error = theory_pre[i] - pred_pre[i]
        max_error = np.max(tmp_error)
        min_error = np.min(tmp_error)
        print(label[i], max_error, min_error)

        plt.grid(True)
        plt.figure()
        plt.plot(test_t, theory_pre[i], label="theory_pre", color="#FF0000")
        plt.plot(test_t, pred_pre[i], '--', label='pred_pre', color="#FF0000")
        plt.xlabel('t')
        plt.ylabel(label[i])
        plt.legend()
        plt.savefig(f"{dir_path}/result{str(idx)}.png")
        idx += 1

        plt.figure()
        plt.plot(test_t, error_percent[i], label="reference_percent", color="#FF0000")
        plt.xlabel('t')
        plt.ylabel(label_percent[i])
        plt.ylim(-100, 100)
        plt.legend()
        plt.savefig(f"{dir_path}/result{str(idx)}.png")
        idx += 1

        plt.figure()
        plt.plot(test_t, error_abs[i], label="precesion_error", color="#FF0000")
        plt.xlabel('t')
        plt.ylabel(label_error_abs[i])
        plt.ylim(-0.1, 0.1)
        plt.legend()
        plt.savefig(f"{dir_path}/result{str(idx)}.png")
        idx += 1
