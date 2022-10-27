# torch.utils.data.DataLoader(): 构建可迭代的数据装载器, 我们在训练的时候，每一个for循环，每一次iteration，就是从DataLoader中获取一个batch_size大小的数据的。
# torch.utils.data.Dataset(): Dataset抽象类， 所有自定义的Dataset都需要继承它，并且必须复写__getitem__()这个类方法。
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

#==========================================step 1/5 准备数据===============================

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


# 数据的路径
split_dir = os.path.join('data', 'rmb_split')
train_dir = os.path.join(split_dir, 'train')
valid_dir = os.path.join(split_dir, 'valid')

## transforms模块，进行数据预处理
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

## 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# print(train_loader)

for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
# rain_loader已经把样本分成了一个个的batch，通过enumerate进行迭代就可以一批批的获取
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # Compute loss
        optimizer.zero_grad()
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # updata weights
        optimizer.step()

