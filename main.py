import random

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import random
from lstm.LSTMModel import LSTMModel


def testFunc(value):
    return math.sin(value) * (1 + (random.random() - 0.5) / 10)


def testFunc1(value):
    return value * value * 0.5 + 1


def testFunc2(value):
    return value * value * value * 0.5 + value * value * 20 + value * 0.4 + 11


def testFunc3(value1, value2):
    return value1 + value2


def testFunc4(value1, value2):
    return math.sqrt(value1) + math.sqrt(value2)


def generateData(seq_len, batch_size):
    data = []
    targets = []
    scale = [random.random() * 4 for i in range(batch_size)]
    for k in range(batch_size):
        cur_data = []
        cur_tar = []
        for i in range(seq_len):
            a = testFunc(i * scale[k])
            b = testFunc1(i * scale[k])
            cur_data.append([i * scale[k], a, b])
            tmp = []
            tmp.append(testFunc3(a, b))
            tmp.append(testFunc4(a, b))
            # tmp.append(testFunc4(testFunc3(a, b), b))
            cur_tar.append(tmp)
        data.append(cur_data)
        targets.append(cur_tar)
    return torch.tensor(data), torch.tensor(targets)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # 定义超参数
    n_outputs = 1
    input_size = 3
    hidden_size = 1
    num_layers = 4
    output_size = 2
    learning_rate = 0.005
    epochs = 500
    batch_size = 200
    seq_len = 10

    # 创建模型实例
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, batch_size, n_outputs)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    inputs, targets = generateData(seq_len, batch_size)

    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        inputs, targets = generateData(seq_len, batch_size)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
    pred_input, pre_tar = generateData(seq_len, 1)
    pred = model(pred_input)
    print(pred_input)
    print(pred)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
