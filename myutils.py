import torchvision.transforms as T
import os
from PIL import Image
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import time
import torch

def print_size(Net):
    param_size = 0
    for param in Net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in Net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def checkpointGenerate(epoch, hparams, Net, name = None):
    model_out_path = hparams["save_folder"]+str(epoch)+ str(name) +"_"+ hparams["model_type"]+".pth".format(epoch)
    torch.save(Net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def print_network(net, hparams):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

    print('===> Building Model ', hparams["model_type"])
    if hparams["model_type"] == 'VQGAN':
        Net = Net


    print('----------------Network architecture----------------')
    print_network(Net)
    print('----------------------------------------------------')


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def save_trainimg(generated_image, epoch, name = ""):
    transform = T.ToPILImage()
    gimg = transform(generated_image.squeeze(0))
    if not os.path.isdir("trainimg"):
        os.mkdir("trainimg")
    gimg.save("trainimg/"+ str(name) + str(epoch) +".png")

def save_allimg(generated_image, label, input, epoch):
    transform = T.ToPILImage()
    gimg = transform(generated_image.squeeze(0))
    lab = transform(label.squeeze(0))
    inp = transform(input.squeeze(0))
    if not os.path.isdir("trainimg"):
        os.mkdir("trainimg")
    gimg.save("trainimg/gen"+ str(epoch) +".png")
    lab.save("trainimg/label"+str(epoch) + ".png")
    inp.save("trainimg/input"+ str(epoch) +".png")


def prepare_imgdatadir(path, outpath, substring = None, numerate = False, startnum = 0, size= None, crop_size = None, multiple_dirs = False):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    all_files = []
    if multiple_dirs is True:
        dirs = os.listdir(path)
        for dir in dirs:
            files = os.listdir(path + "/" + dir)
            outfiles = []
            if substring != None:
                for file in files:
                    if substring in file:
                        outfiles.append(file)
            for file in outfiles:
                all_files.append(dir + "/" + file)
    else:
        files = os.listdir(path)
        outfiles = []
        if substring != None:
            for file in files:
                if substring in file:
                    outfiles.append(file)
        else:
            outfiles = files

        all_files = outfiles


    for i,file in enumerate(BackgroundGenerator(tqdm(all_files), max_prefetch=5), start=startnum):
            img = Image.open(path + "/" + file)
            if size != None:
                img = img.resize(size)
            if crop_size != None:
                img = crop_center(img, crop_size[0], crop_size[1])
            if numerate:
                img.save(outpath + str(i) + ".png")
            else:
                img.save(outpath + file)

if __name__ == "__main__":
    prepare_imgdatadir("C:/Data/dehaze/SOTS/indoor/gt", "C:/Data/dehaze/test/gt/" ,substring = None ,numerate=True,startnum = 0, size=(1920,1080), multiple_dirs=False)