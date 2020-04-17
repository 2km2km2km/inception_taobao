#各种库的导入
from __future__ import division
from torchvision.datasets import ImageFolder
from models import *
#from utils.utils import *
from utils.datasets import *
from utils.loss import *
from utils.parse_config import *
from utils.get_txt import *
#from test import evaluate
from test import *


#from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

w=SummaryWriter(comment="inception")

def loadcap(batch_size):  # 图片的大小是64*64
    trans_img = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    trainset = ImageFolder('./data/train/', transform=trans_img)
    testset = ImageFolder('./data/test/', transform=trans_img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    return trainset, testset, trainloader, testloader

def train(epochs):
    #载入部分超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/casia.data", help="path to data config file")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    opt = parser.parse_args()

    trainset, testset, trainloader, testloader = loadcap(5)
    #创建文件夹
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    #判断能否使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化网络模型
    model = InceptionV2().to(device)

    # 获取数据配置
    data_config = parse_data_config(opt.data_config)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    precisions=[]
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(trainloader), batch_time, losses, top1,top5)
        epoch_loss=0
        for batch_i, (imgs, y_reals) in enumerate(trainloader):
            if batch_i>5:
                break
            batches_done = len(trainloader) * epoch + batch_i  #已训练的图片batch数
            imgs = Variable(imgs.to(device))
            y_reals = Variable(y_reals.to(device), requires_grad=False)
            y_hats = model(imgs)
            loss=loss_CEL(y_hats,y_reals)
            acc1, acc5 = accuracy(y_hats, y_reals, topk=(1, 2))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))
            loss.backward()
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            #打印信息
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(trainloader))
            print(log_str)
            log_str += f"\nTotal loss {loss.item()}"
            epoch_batches_left = len(trainloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)
            epoch_loss+=loss.item()
        print("epoch",epoch)
        print("loss",epoch_loss)
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            val_losses,val_top1,val_top5=evaluate(
                model,
                testloader,
                opt.img_size,
                6,
                loss_CEL
            )
        #保存权重
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/inception_ckpt_%d.pth" % epoch)
        #写入可视化
        w.add_scalars("loss",{"train":losses.avg,"val":val_losses.avg},epoch)
        w.add_scalars("top1",{"train":top1.avg,"val":val_top1.avg},epoch)
        w.add_scalars("top5",{"train":top5.avg,"val":val_top5.avg},epoch)
    w.close()
    return precisions

if __name__ == "__main__":
    train(300)
