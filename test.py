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

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.linear1(x)
        return x[-1, :, :]


def generateData(len, sample_len):
    data_x = []
    data_y = []
    step = 100
    tmp_data = []
    for i in range(len + sample_len):
        tmp_data.append(math.sin(i * math.pi / step) * (1 + (random.random() - 0.5) / 5))

    for i in range(len):
        data_x.append(tmp_data[i:i + sample_len])
        data_y.append([tmp_data[i + sample_len]])

    data_x = np.array(data_x).astype('float32')
    data_y = np.array(data_y).astype('float32')
    return data_x, data_y


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
    sample_len = 8

    data_x, data_y = generateData(2000, sample_len)

    # 数据集分割
    data_len = len(data_x)
    t = np.linspace(0, data_len - 1, data_len)

    train_data_ratio = 0.5  # Choose 80% of the data for training
    train_data_len = int(data_len * train_data_ratio)

    train_x = data_x[:train_data_len]
    train_y = data_y[:train_data_len]
    t_for_training = t[:train_data_len]

    test_x = data_x[train_data_len - 1:]
    test_y = data_y[train_data_len - 1:]
    t_for_testing = t[train_data_len - 1:]

    # ----------------- train -------------------
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    train_x_tensor = train_x.reshape(sample_len, -1, INPUT_FEATURES_NUM)
    train_y_tensor = train_y.reshape(-1, OUTPUT_FEATURES_NUM)
    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 30, output_size=OUTPUT_FEATURES_NUM, num_layers=5)  # 20 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    print('train x tensor dimension:', Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.003)

    prev_loss = 1000
    max_epochs = 4000

    train_x_tensor = train_x_tensor.to(device)

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
    pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- test -------------------
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(sample_len, -1, INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = lstm_model(test_x_tensor).to(device)
    pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
    print("test loss：", loss.item())

    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_y, 'b', label='y_trn')
    plt.plot(t_for_training, pred_y_for_train, 'y--', label='pre_trn')

    plt.plot(t_for_testing, test_y, 'k', label='y_tst')
    plt.plot(t_for_testing, pred_y_for_test, 'm--', label='pre_tst')

    plt.xlabel('t')
    plt.ylabel('Vce')
    plt.show()
