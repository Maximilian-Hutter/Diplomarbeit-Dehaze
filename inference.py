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
parser.add_argument('--modelpath', type=str, default="weights/9cityscapes_Dehaze.pth", help=("path to the model .pth files"))
parser.add_argument('--inferencepath', type=str, default='C:/Data/dehaze/inference/', help=("Path to image folder"))
parser.add_argument('--imagename', type=str, default='foggy.jpg', help=("filename of the image"))
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
    image = image.resize((int(hparams["height"]), int(hparams["width"])))
    image.save('results/foggy.png')

    transformtotensor = transforms.Compose([transforms.ToTensor()])
    image = transformtotensor(image)

    image = image.unsqueeze(0)

    image= image.to(torch.float32)

    model=Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"], hparams["down_deep"], pseudo_alpha= hparams["pseudo_alpha"], hazy_alpha=hparams["hazy_alpha"])

    if hparams["gpu_mode"] == False:
        device = torch.device('cpu')

    if hparams["gpu_mode"]:
            device = torch.device('cuda')

    model.load_state_dict(torch.load(PATH,map_location=device))

    model.eval()
    model = model.cuda()
    start = time.time()
    transform = T.ToPILImage()
    image = image.to(torch.device('cuda'))
    times = []
    allproctime = 0
    out, pseudo = model(image)
    for i in range(100):
        start = time.time()
        out, pseudo = model(image)
        end = time.time()
        proctime = round(end -start, 4)
        allproctime += proctime

    times = allproctime/100
    print(times)
    out = transform(out.squeeze(0))
    out.save('results/no_fog.png')
    pseudo = transform(pseudo.squeeze(0))
    pseudo.save('results/pseudo.png')