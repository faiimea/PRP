import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class faii(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        self.linear1=Linear(196608,10)

    def forward(self,x):
        #x=self.conv1(x)
        x = self.linear1(x)
        return x

dataset=torchvision.datasets.CIFAR10(root="tv/dataset",train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64)

ff=faii()
step=0
writer=SummaryWriter("./logs")
for data in dataloader:
    imgs,target=data
    # output=torch.reshape(imgs,(1,1,1,-1))
    output=torch.flatten(imgs)
    output=ff(output)

    print(output.shape)
    # output=torch.reshape(output,(-1,3,30,30))
    # writer.add_images("input",imgs,step)
    # writer.add_images("output",output,step)
    # step=step+1