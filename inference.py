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
    hparams = {
        "seed": 123,
        "gpus": 1,
        "gpu_mode": True,
        "crop_size": None,
        "resume": False,
        "train_data_path": "C:/Data/dehaze/prepared/",
        "augment_data": False,
        "epochs_o_haze": 10,
        "epochs_nh_haze": 10,
        "epochs_cityscapes": 30,
        "batch_size": 1,
        "crit_lambda": 1,
        "threads": 4,
        "height":256,
        "width":256,
        "lr": 0.0004,
        "beta1": 0.9,
        "beta2": 0.999,
        "mhac_filter": 256,
        "mha_filter": 16,
        "num_mhablock": 4,
        "num_mhac": 3,
        "num_parallel_conv": 2,
        "kernel_list": [3,5,7],
        "pad_list": [4,12,24],
        "save_folder": "./weights/",
        "model_type": "Dehaze",
        "snapshots": 10
    }

    opt = parser.parse_args()

    PATH = opt.modelpath
    imagepath = (opt.inferencepath + opt.imagename)
    image = Image.open(imagepath)
    image = image.resize((int(256/opt.scale), int(256/opt.scale)))
    image.save('results/foggy.png')

    transformtotensor = transforms.Compose([transforms.ToTensor()])
    image = transformtotensor(image)

    image = image.unsqueeze(0)

    image= image.to(torch.float32)

    model=Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"]).cuda()

    if opt.gpu_mode == False:
        device = torch.device('cpu')

    if opt.gpu_mode:
            device = torch.device('cuda:0')

    model.load_state_dict(torch.load(PATH,map_location=device))

    model.eval()
    start = time.time()
    transform = T.ToPILImage()
    out = model(image)
    out = transform(out.squeeze(0))
    end = time.time()
    proctime = end -start
    out.save('results/no_fog.png')
    print(proctime)