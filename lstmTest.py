import random

import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.autograd import Variable
import math
import motion.motionDataMoc
import csv


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层
        # self.activation = nn.ReLU()

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.linear1(x)
        # out = self.activation(x[-1, :, :])
        out = x[-1, :, :]
        return out


def testFunc(value):
    return math.sin(value) * (1 + (random.random() - 0.5) / 10)


def testFunc1(value):
    ret = value * value * 0.2 + 1
    ret = ret * (1 + (random.random() - 0.5) / 10)
    return ret


def testFunc2(value):
    ret = value * value * value * 0.01 + value * value * 0.3 + value * 0.4 + 11
    ret = ret * (1 + (random.random() - 0.5) / 10)
    return ret


def testFunc3(value1, value2):
    return value1 * value1 + value1 * value2


def testFunc4(value1, value2):
    return math.sqrt(value1) + math.sqrt(value2)


def generateData(len, sample_len):
    data_x = []
    data_y = []
    step = 80
    tmp_data = []
    tmp_data1 = []
    next_step = 8
    for i in range(len + sample_len + next_step):
        # a = testFunc((i - 1) * math.pi / step)
        a = testFunc2((i) / step)
        b = testFunc1((i) / step)
        c = testFunc4(a, b)
        tmp_data.append([a, b])
        tmp_data1.append([c, c])

    for i in range(len):
        data_x.append(tmp_data[i:i + sample_len])
        data_y.append([tmp_data1[i + sample_len + next_step]])

    data_x = np.array(data_x).astype('float32')
    data_y = np.array(data_y).astype('float32')
    return data_x, data_y


# lstm训练模型
if __name__ == '__main__':

    # checking if GPU is available
    device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    # 数据读取&类型转换
    # data_x = np.array(pd.read_csv('Data_x.csv', header=None)).astype('float32')
    # data_y = np.array(pd.read_csv('Data_y.csv', header=None)).astype('float32')
    sample_len = 1

    # data_x, data_y = generateData(500, sample_len)

    # 生成训练数据
    data_x, data_y, error_value = motion.motionDataMoc.generateData(500, sample_len, [0.1, 0.15, 0.2, 0.25, 0.3])
    # data_x, data_y, error_value = motion.motionDataMoc.generateData(500, sample_len, None)
    # 生成测试数据
    test_x, test_y, test_error_value = motion.motionDataMoc.generateData(500, sample_len, [0.1, 0.15, 0.2, 0.25, 0.3],
                                                                         [0.5, 1, 1])
    # 数据集分割
    data_len = len(data_x)
    t = np.linspace(0, data_len - 1, data_len)

    train_data_ratio = 1.0  # Choose 80% of the data for training
    train_data_len = int(data_len * train_data_ratio)

    train_x = data_x[:train_data_len]
    train_y = data_y[:train_data_len]
    t_for_training = t[:train_data_len]

    # test_x = data_x[train_data_len - 1:]
    # test_y = data_y[train_data_len - 1:]
    # t_for_testing = t[train_data_len - 1:]
    t_for_testing = t

    # ----------------- train -------------------
    INPUT_FEATURES_NUM = data_x.shape[-1]
    OUTPUT_FEATURES_NUM = data_y.shape[-1]
    train_x_tensor = train_x.reshape(sample_len, -1, INPUT_FEATURES_NUM)
    train_y_tensor = train_y.reshape(-1, OUTPUT_FEATURES_NUM)
    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 10, output_size=OUTPUT_FEATURES_NUM, num_layers=5)  # 20 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    print('train x tensor dimension:', Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0005)

    prev_loss = 1000
    max_epochs = 1600

    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor).to(device)
        loss = criterion(output, train_y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < prev_loss:
            torch.save(lstm_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
            prev_loss = loss

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # prediction on training dataset
    pred_y_for_train = lstm_model(train_x_tensor).to(device)
    pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()

    # ----------------- test -------------------
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(sample_len, -1, INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = lstm_model(test_x_tensor).to(device)
    pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
    print("test loss：", loss.item())

    # ----------------- plot -------------------
    plt.figure()
    color = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#000000', '#880000', '#008800',
             '#000088']
    for i in range(train_y.shape[2]):
        # train_y_i = train_y[:, :, i].reshape(-1)
        # pred_y_for_train_i = pred_y_for_train[:, i].reshape(-1)
        # plt.plot(t_for_training, train_y_i, 'b', label='y_trn' + str(i))
        # plt.plot(t_for_training, pred_y_for_train_i, 'y--', label='pre_trn' + str(i))

        test_y_i = test_y[:, :, i].reshape(-1)
        pred_y_for_test_i = pred_y_for_test[:, i].reshape(-1)
        plt.plot(t_for_testing, test_y_i, 'k', label='y_tst' + str(i), color=color[i])
        plt.plot(t_for_testing, pred_y_for_test_i, 'm--', label='pre_tst' + str(i), color=color[i])

    plt.xlabel('test_x')
    plt.ylabel('text_y')

    plt.figure()
    for i in range(train_y.shape[2]):
        train_y_i = train_y[:, :, i].reshape(-1)
        pred_y_for_train_i = pred_y_for_train[:, i].reshape(-1)
        plt.plot(t_for_training, train_y_i, 'b', label='y_trn' + str(i), color=color[i])
        plt.plot(t_for_training, pred_y_for_train_i, 'y--', label='pre_trn' + str(i), color=color[i])

    plt.xlabel('train_x')
    plt.ylabel('train_y')

    plt.figure()
    for i in range(4):
        plt.plot(t, error_value[:, i], 'b', label='input')
    plt.xlabel('t')
    plt.ylabel('input')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(error_value[:, 4], error_value[:, 5], error_value[:, 6])
    ax.plot(test_error_value[:, 4], test_error_value[:, 5], test_error_value[:, 6])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # for i in range(error_value.shape[-1]):
    #     plt.figure()
    #     plt.plot(t, error_value[:, i], 'b', label='input' + str(i))
    #     plt.xlabel('t')
    #     plt.ylabel('input')

    plt.show()
