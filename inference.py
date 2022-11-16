import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor
from models import *
import torchvision.transforms as T
from torchvision import transforms, utils
from PIL import Image
import argparse
import time
import torchvision
from params import hparams

parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
parser.add_argument('--modelpath', type=str, default="weights/9Dehaze.pth", help=("path to the model .pth files"))
parser.add_argument('--inferencepath', type=str, default='C:/Data/dehaze/test/', help=("Path to image folder"))
parser.add_argument('--imagename', type=str, default='foggy.png', help=("filename of the image"))
parser.add_argument('--gpu_mode', type=bool, default=True, help=('enable cuda'))
parser.add_argument('--channels',type=int, default=3, help='number of channels R,G,B for img / number of input dimensions 3 times 2dConv for img')
parser.add_argument('--filters', type=int, default=8, help="set number of filters")
parser.add_argument('--bottleneck', type=int, default=8, help="set number of filters")
parser.add_argument('--n_resblock', type=int, default=3, help="set number of filters")
parser.add_argument('--scale', type=int, default=2, help="set number of filters")
    
if __name__ == '__main__':


    opt = parser.parse_args()

    PATH = opt.modelpath
    imagepath = (opt.inferencepath + opt.imagename)
    image = Image.open(imagepath)
    image = image.resize((int(hparams["height"]/opt.scale), int(hparams["width"]/opt.scale)))
    image.save('results/foggy.png')

    transformtotensor = transforms.Compose([transforms.ToTensor()])
    image = transformtotensor(image)

    image = image.unsqueeze(0)

    image= image.to(torch.float32)

    model=Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"])

    if hparams["gpu_mode"] == False:
        device = torch.device('cpu')

    if hparams["gpu_mode"]:
            device = torch.device('cuda')

    model.load_state_dict(torch.load(PATH,map_location=device))

    model.eval()
    start = time.time()
    transform = T.ToPILImage()
    image = image.to(torch.device('cuda'))
    times = []
    for i in range(100):
        start = time.time()
        out = model(image)
        end = time.time()
        proctime = round(end -start, 4)
        if proctime <= 0.0300:
            print(proctime)

    out = out[0]
    out = transform(out.squeeze(0))
    
    out.save('results/no_fog.png')