import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MaxPool2d
from torch.nn import Flatten
from faii_nn import *
# 数据预处理，基于CIFAR10数据集

train_data=torchvision.datasets.CIFAR10(root="tv/dataset",train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="tv/dataset",train=False, transform=torchvision.transforms.ToTensor(),download=True)
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)
test_data_size=len(test_data)
train_data_size=len(train_data)
# 十分类神经网络
ff=faii_nn()
loss_function=nn.CrossEntropyLoss()

# 优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(ff.parameters(),lr=learning_rate)

# 设置训练网络参数
total_train_step=0
total_test_step=0
epoch=30

# board
writer=SummaryWriter("logs_nm")

for i in range(epoch):
     print("第{}轮训练开始".format(i+1))

     for data in train_dataloader:
          imgs,targets=data
          outputs=ff(imgs)
          loss=loss_function(outputs,targets)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          total_train_step+=1
          if total_train_step%100==0:
               print("train_time={},loss={}".format(total_train_step, loss))
               writer.add_scalar("train_loss",loss.item(),total_train_step)

     total_test_loss=0
     # acc
     total_accuracy=0
     with torch.no_grad():
          for data in test_dataloader:
               imgs,targets=data
               outputs=ff(imgs)
               loss=loss_function(outputs,targets)
               total_test_loss+=loss
               accuracy=(outputs.argmax(1)==targets).sum()
               total_accuracy=total_accuracy+accuracy

     writer.add_scalar("test_loss", total_test_loss.item(), total_test_step)
     print("total_test_loss={}".format(total_test_loss),total_test_step)
     print("total_accuracy={}".format(total_accuracy/test_data_size))
     writer.add_scalar("total_accuracy", (total_accuracy/test_data_size), total_test_step)
     total_test_step+=1

     torch.save(ff,"ff_nmodel/ff_nm_{}.pth".format(i))
     print("model has been saved")


writer.close()

# print(faii)
# input=torch.ones((64,3,32,32))
# output=ff(input)
# print(output.shape)

# step=0
# writer=SummaryWriter("./logs_seq")
# writer.add_graph(ff,input)
# writer.close()

    #  output=torch.reshape(imgs,(1,1,1,-1))
    #  output=torch.flatten(imgs)
    #  output=ff(output)
    #
    # print(output.shape)
    # output=torch.reshape(output,(-1,3,30,30))
    # writer.add_images("input",imgs,step)
    # writer.add_images("output",output,step)
    # step=step+1