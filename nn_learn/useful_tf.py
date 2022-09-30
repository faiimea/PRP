from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")

img=Image.open("images/22969fa72ef7bfe03bf3fe5aee2a5cf.jpg")

print(img)

trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)

print(img_tensor[1][1][0])
trans_norm = transforms.Normalize([1,3,4],[2,3,1])
img_norm=trans_norm(img_tensor)
print(img_norm[1][1][0])

trans_resize=transforms.Resize((514,514))
img_resize=trans_resize(img)
img_resize=trans_totensor(img_resize)

trans_resize_2=transforms.Resize(514)

# compose实现流水线，数据需为transforms类型

trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)

writer.add_image("totensor1",img_norm)
writer.add_image("totensor2",img_resize)

trans_random=transforms.RandomCrop(114)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("rc",img_crop,i)


writer.close()