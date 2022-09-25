import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True, transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False, transform=dataset_transform,download=True)
test_loader=DataLoader(test_set,batch_size=4,shuffle=True,num_workers=0,drop_last=False)


for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)

# print(test_set)

# writer= SummaryWriter("p10")
# for i in range(10):
#     img,target=test_set[i]
#     writer.add_image("test_set",img,i)

# writer.close()