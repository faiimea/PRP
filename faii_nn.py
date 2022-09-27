import torch
from torch import nn
from torch.nn import Sequential, Conv2d,MaxPool2d,Flatten,Linear


class faii_nn(nn.Module):
    def __init__(self):
        super(faii_nn,self).__init__()
        # self.conv1=Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
        # self.maxpool1=MaxPool2d(2)
        # self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten=Flatten()
        # self.linear1=Linear(1024,64)
        # self.linear2 = Linear(64, 10)
        self.model1=Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x=self.model1(x)
        return x


if __name__=='__main__':
    ff=faii_nn()
    input=torch.ones((64,3,32,32))
    output=ff(input)
    print(output.shape)