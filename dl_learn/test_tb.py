from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


writer = SummaryWriter("./logs")
image_path="dataset/train/ants/0013035.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
writer.add_image("test",img_array,1,dataformats='HWC')
# y=x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()