import torch.nn as nn
import torch
loss_CEL = nn.CrossEntropyLoss()  # 定义损失函数
loss_MSE=nn.MSELoss()
def one_hot(y_real,class_num):
    y=torch.zeros((y_real.size(0),class_num))
    for i,j in enumerate(y_real):
        y[i,j]=1
    return y

