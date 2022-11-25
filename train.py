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
from torch.utils.tensorboard import SummaryWriter

# my imports
import myutils
from loss import ChabonnierLoss
from params import hparams

if __name__ == '__main__':


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
    dataloader_o_haze = DataLoader(ImageDataset(hparams["train_data_path"] + "O-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_nh_haze = DataLoader(ImageDataset(hparams["train_data_path"] + "NH-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_cityscapes = DataLoader(ImageDataset(hparams["train_data_path"] + "cityscapes",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])

    # define the Network
    Net = Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"], hparams["gpu_mode"], hparams["scale_factor"])

    # print Network parameters
    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    # set criterions & optimizers
    criterion = ChabonnierLoss(eps = 1e-6)
    l1_criterion = nn.L1Loss()
    optimizer = optim.Adam(Net.parameters(), lr=hparams["lr"], betas=(hparams["beta1"],hparams["beta2"]))

    cuda = hparams["gpu_mode"]
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


    torch.manual_seed(hparams["seed"])
    if cuda:
        torch.cuda.manual_seed(hparams["seed"])

    if cuda:
        Net = Net.to(torch.device("cuda:0"))
        criterion = criterion.to(torch.device("cuda:0"))
        l1_criterion = l1_criterion.to(torch.device("cuda:0"))

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

    #Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # performance optimizations
    Net = Net.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
    #torch.autograd.profiler.emit_nvtx(enabled=False)
    #torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter()

    
    print("Starting Training")

    for epoch in range(start_epoch, hparams["epochs_nh_haze"]):
        epoch_loss = 0
        Net.train()
        epoch_time = time.time()
        correct = 0

        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader_nh_haze)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            start_time = time.time()
            img = Variable(imgs["img"])
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"])

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                label = label.to(gpus_list[0])

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image, pseudo = Net(img)
                chabonnier_gen = criterion(generated_image, label)
                chabonnier_pseudo = criterion(pseudo, label)
                loss = hparams["gen_lambda"] * chabonnier_gen + hparams["pseudo_lambda"] * chabonnier_pseudo
       
            if hparams["batch_size"] == 1:
                if i == 1:
                    myutils.save_trainimg(pseudo, epoch, "pseudo_nh_haze")
                    myutils.save_trainimg(generated_image, epoch, "generated_nh_haze")

            train_acc = torch.sum(generated_image == label)
            epoch_loss += loss.item()

            if cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


        if (epoch+1) % (hparams["snapshots"]) == 0:
            myutils.checkpointGenerate(epoch, hparams, Net, "nh_haze_")


        epoch_time = time.time() - epoch_time 
        Accuracy = 100*train_acc / len(dataloader_nh_haze)
        writer.add_scalar('loss', epoch_loss, global_step=epoch)
        writer.add_scalar('accuracy',Accuracy, global_step=epoch)
        print("===> Epoch {} Complete: Avg. loss: {:.4f} Accuracy {}, Epoch Time: {:.3f} seconds".format(epoch, ((epoch_loss/2) / len(dataloader_nh_haze)), Accuracy, epoch_time))
        print("\n")
        epoch_time = time.time()

    # pretrain in O Haze
    for epoch in range(start_epoch, hparams["epochs_o_haze"]):
        epoch_loss = 0
        Net.train()
        epoch_time = time.time()
        correct = 0

        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader_o_haze)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            start_time = time.time()
            img = Variable(imgs["img"])
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"])


            img = img.to(torch.device('cuda'))
            label = label.to(torch.device('cuda'))

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image, pseudo = Net(img)
                chabonnier_gen = criterion(generated_image, label)
                chabonnier_pseudo = criterion(pseudo, label)
                loss = hparams["gen_lambda"] * chabonnier_gen + hparams["pseudo_lambda"] * chabonnier_pseudo
            
            if hparams["batch_size"] == 1:
                if i == 1:
                    myutils.save_trainimg(pseudo, epoch, "pseudo_o-haze")
                    myutils.save_trainimg(generated_image, epoch, "generated_o_haze")

            train_acc = torch.sum(generated_image == label)
            epoch_loss += loss

            scaler.scale(loss).backward() #derivative for channel_shuffle is not implemented
            scaler.step(optimizer)
            scaler.update()

            #compute time and compute efficiency and print information
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


        if (epoch+1) % (hparams["snapshots"]) == 0:
            myutils.checkpointGenerate(epoch, hparams, Net, "o_haze_")


        epoch_time = time.time() - epoch_time 
        Accuracy = 100*train_acc / len(dataloader_o_haze)
        writer.add_scalar('loss', epoch_loss, global_step=epoch)
        writer.add_scalar('accuracy',Accuracy, global_step=epoch)
        print("===> Epoch {} Complete: Avg. loss: {:.4f} Accuracy {}, Epoch Time: {:.3f} seconds".format(epoch, ((epoch_loss/2) / len(dataloader_o_haze)), Accuracy, epoch_time))
        print("\n")
        epoch_time = time.time()



    for epoch in range(start_epoch, hparams["epochs_cityscapes"]):
        epoch_loss = 0
        Net.train()
        epoch_time = time.time()
        correct = 0

        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader_cityscapes)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            start_time = time.time()
            img = Variable(imgs["img"])
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"])

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                label = label.to(gpus_list[0])

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image, pseudo = Net(img)
                chabonnier_gen = criterion(generated_image, label)
                chabonnier_pseudo = criterion(pseudo, label) 
                loss = hparams["gen_lambda"] * chabonnier_gen + hparams["pseudo_lambda"] * chabonnier_pseudo
            
            
            if hparams["batch_size"] == 1:
                if i == 1:
                    myutils.save_trainimg(pseudo, epoch, "pseudo_cityscapes")
                    myutils.save_trainimg(generated_image, epoch, "generated_cityscapes")

            train_acc = torch.sum(generated_image == label)
            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


        if (epoch+1) % (hparams["snapshots"]) == 0:
            myutils.checkpointGenerate(epoch, hparams, Net, "cityscapes_")

        epoch_time = time.time() - epoch_time 
        Accuracy = 100*train_acc / len(dataloader_cityscapes)
        writer.add_scalar('loss', epoch_loss, global_step=epoch)
        writer.add_scalar('accuracy',Accuracy, global_step=epoch)
        print("===> Epoch {} Complete: Avg. loss: {:.4f} Accuracy {}, Epoch Time: {:.3f} seconds".format(epoch, ((epoch_loss/2) / len(dataloader_cityscapes)), Accuracy, epoch_time))
        print("\n")
        epoch_time = time.time()



    myutils.print_network(Net, hparams)