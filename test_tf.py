from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# transform.usage & tensor & why

img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

tensor_trans=transforms.ToTensor()  # __init__ create specific tool

# cv2.imread(img_path) -> ndarray
# Image.open(img_path) -> PIL Image
# INPUT of ToTensor can be such types

tensor_img=tensor_trans(img)    # __call__ use it
writer.add_image("Tensor_img",tensor_img)

writer.close()