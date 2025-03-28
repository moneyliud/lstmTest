import torch.optim
import torch.nn as nn
import torch
from transformer.TFMPrecisionModel import TransformerLM
from tqdm import tqdm
from random import random


# TFM网络模型训练代码
class TFMPrecision:
    def __init__(self, data, seq_length=100, epoch=-1, batch_size=1, lr=0.001, loss_thres=0.001, model_path=None,
                 weight_path=None):
        # 查询是否能使用gpu进行训练
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 模型文件名称
        self.model_file_name = "TFMP_"
        # 最大迭代次数
        self.epoch = epoch
        # 损失值阈值，达到该阈值停止计算
        self.loss_thres = loss_thres
        # 输入的序列长度
        self.seq_len = seq_length
        # 输入层个点位嵌入向量大小
        self.embedding_size = data[0].shape[-1]
        # 输出层参数个数
        self.output_size = data[1].shape[-1]
        # 输入数据
        self.data_x = torch.tensor(data[0], dtype=torch.double, device=self.device)
        # 用于输出的判别数据
        self.data_y = torch.tensor(data[1], dtype=torch.double, device=self.device)
        # 数据长度
        self.total_len = self.data_x.shape[0]
        # 训练的批次大小
        self.batch_size = min(batch_size, self.total_len)
        torch.set_default_tensor_type(torch.DoubleTensor)
        if model_path is not None and weight_path is not None:
            model = torch.load(model_path)
            model.load_state_dict(torch.load(weight_path), device=self.device)
            # params = model.named_parameters()
            # for name, layer in params:
            #     if not name.find("last_layer"):
            #         layer.requires_grad = False
            # print(model.last_layer)
            self.model = model
        else:
            self.model = TransformerLM(self.seq_len, self.embedding_size, self.output_size, num_layers=16,
                                       num_heads=4)
        self.model = self.model.to(self.device)
        self.model.double()
        # 学习率
        self.lr = lr
        self.data = []
        # for i in range(len(self.raw_data)):
        #     self.data.append(torch.tensor(self.raw_data[i].to_array()).unsqueeze(0).unsqueeze(0))
        # self.data = torch.tensor(self.data).unsqueeze(1).unsqueeze(1)
        # mean = torch.mean(self.data, axis=0)
        # std = torch.std(self.data, axis=0)
        # data1 = (self.data - std) / mean
        # self.data = data1.unsqueeze(1).unsqueeze(1)
        pass

    # 模型训练函数
    def run(self):
        # 设置迭代训练优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # 设置损失函数
        loss_func = nn.L1Loss()
        # 保存模型文件
        torch.save(self.model, self.model_file_name + "model.pt")
        best_loss = None

        # 开始迭代训练
        for times in range(self.epoch):
            avgloss = 0.0
            n = int(self.total_len / self.batch_size) if self.total_len % self.batch_size == 0 else (
                    int(self.total_len / self.batch_size) + 1)
            batch_x, batch_y = self.data_x, self.data_y
            mask = torch.triu(torch.ones((self.seq_len, self.seq_len)), diagonal=1).bool()
            # 遍历输入数据
            for i in tqdm(range(n)):
                # 批次起始位置
                start = i * self.batch_size
                # 批次结束位置
                end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < self.total_len else self.total_len
                # 损失值初始为0
                tmploss = torch.zeros(1, device=self.device)
                batch_data = batch_x[start:end]
                pred = self.model(batch_data, mask)
                target = batch_y[start:end]
                # 计算损失值
                loss1 = loss_func(pred, target) * 100
                loss = tmploss + loss1
                # 计算
                avgloss += loss
                # 梯度清零
                optimizer.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 优化器根据梯度进行神经网络参数更新
                optimizer.step()
            # 计算该次训练的平均损失
            avgloss = avgloss / n
            if best_loss is None:
                best_loss = avgloss
            elif avgloss < best_loss:
                best_loss = avgloss
                # 保存最佳模型
                torch.save(self.model.state_dict(), self.model_file_name + 'best.pt')
            print(str(times + 1) + "/" + str(self.epoch), ":", "avgloss=", avgloss, "\tbestloss = ",
                  best_loss)
            # 保存模型文件
            torch.save(self.model.state_dict(), self.model_file_name + "params.pt")
            if avgloss < self.loss_thres:
                break
