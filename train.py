from data import ImageDataset
from models import *
import numpy as np
import torch
import torch.nn as nn
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torch.autograd import Variable

# my imports
import myutils

if __name__ == '__main__':
    hparams = {
        "seed": 123,
        "gpus": 1,
        "gpu_mode": True,
        "crop_size": None,
        "train_data_path": "C:/Data/dehaze/prepared/",
        "augment_data": False,
        "epochs_haze": 100,
        "epochs_frida": 50,
        "epochs_cityscape": 300,
        "batch_size": 1,
        "threads": 4,
        "height":144,
        "width":144,
        "lr": 0.0004,
        "beta1": 0.9,
        "beta2": 0.999
    }

    np.random.seed(hparams["seed"])    # set seed to default 123 or opt
    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed(hparams["seed"])
    gpus_list = range(hparams["gpus"])
    cuda = hparams["gpu_mode"]
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(hparams)

    size = (hparams["height"], hparams["width"])

    print('==> Loading Datasets')
    dataloader_frida = DataLoader(ImageDataset(hparams["train_data_path"] + "frida/",size,hparams["crop_size"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_nh_haze = DataLoader(ImageDataset(hparams["train_data_path"] + "NH_Haze/",size,hparams["crop_size"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_cityscapes = DataLoader(ImageDataset(hparams["train_data_path"] + "cityscapes/",size,hparams["crop_size"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])

    # define the Network
    Net = dehaze()

    # print Network parameters
    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    # set criterions & optimizers
    criterion = torch.nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(Net.parameters(), lr=hparams["lr"], betas=(hparams["beta1"],hparams["beta2"]))

    myutils.setcuda(hparams, gpus_list)

    # load checkpoint/load model
    star_n_iter = 0
    start_epoch = 0
    if hparams["resume"]:
        checkpoint = torch.load(hparams["save_folder"]) ## look at what to load
        start_epoch = checkpoint['epoch']
        start_n_iter = checkpoint['n_iter']
        optimizer.load_state_dict(checkpoint['optim'])
        print("last checkpoint restored")

    # define Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # performance optimizations
    Net = Net.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    # pretrain in NH_HAZE
    for epoch in range(start_epoch, hparams["epochs_haze"]):
        epoch_loss = 0
        Net.train()
        epoch_time = time.time()
        correct = 0

        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader_nh_haze)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            start_time = time.time()
            img = Variable(imgs["img"].type(Tensor))
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"].type(Tensor))

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                label = label.to(gpus_list[0])

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image = Net(img)
                crit = criterion(generated_image, label)
                loss = hparams["crit_lambda"] *  crit
            
            if hparams["batch_size"] == 1:
                if i == 1:
                    myutils.save_trainimg(generated_image, epoch)

            train_acc = torch.sum(generated_image == label)
            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


        if (epoch+1) % (hparams["snapshots"]) == 0:
            myutils.checkpointGenerate(epoch, hparams, Net)

        myutils.print_info(epoch, epoch_loss,train_acc, dataloader_nh_haze)



    for epoch in range(start_epoch, hparams["epochs_frida"]):
        epoch_loss = 0
        Net.train()
        epoch_time = time.time()
        correct = 0

        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader_frida)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            start_time = time.time()
            img = Variable(imgs["img"].type(Tensor))
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"].type(Tensor))

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                label = label.to(gpus_list[0])

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image = Net(img)
                crit = criterion(generated_image, label)
                loss = hparams["crit_lambda"] *  crit
            
            if hparams["batch_size"] == 1:
                if i == 1:
                    myutils.save_trainimg(generated_image, epoch)

            train_acc = torch.sum(generated_image == label)
            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


        if (epoch+1) % (hparams["snapshots"]) == 0:
            myutils.checkpointGenerate(epoch, hparams, Net)

        myutils.print_info(epoch, epoch_loss,train_acc, dataloader_frida)

    
    for epoch in range(start_epoch, hparams["epochs_cityscapes"]):
        epoch_loss = 0
        Net.train()
        epoch_time = time.time()
        correct = 0

        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader_cityscapes)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            start_time = time.time()
            img = Variable(imgs["img"].type(Tensor))
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"].type(Tensor))

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                label = label.to(gpus_list[0])

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image = Net(img)
                crit = criterion(generated_image, label)
                loss = hparams["crit_lambda"] *  crit
            
            if hparams["batch_size"] == 1:
                if i == 1:
                    myutils.save_trainimg(generated_image, epoch)

            train_acc = torch.sum(generated_image == label)
            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


        if (epoch+1) % (hparams["snapshots"]) == 0:
            myutils.checkpointGenerate(epoch, hparams, Net)

        myutils.print_info(epoch, epoch_loss,train_acc, dataloader_cityscapes)



myutils.print_network(Net, hparams)