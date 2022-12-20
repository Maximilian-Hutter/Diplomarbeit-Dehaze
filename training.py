import time
import torch
import torch.nn
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import myutils
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from models import Dehaze

class Train():
    def __init__(self, hparams, Net, optimizer, criterion) :
    
        Net = Net.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.backends.cudnn.benchmark = True
        self.scaler = torch.cuda.amp.GradScaler()
        self.writer = SummaryWriter()

        self.criterion = criterion
        self.gpus_list = range(hparams["gpus"])
        self.hparams = hparams
        self.Net = Net
        self.optimizer = optimizer

    def train(self, dataloader,datasetname, epochs):
        for epoch in range(self.hparams["start_epoch"], epochs):
            epoch_loss = 0
            self.Net.train()
            epoch_time = time.time()
            correct = 0

            for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

                start_time = time.time()
                img = Variable(imgs["img"])
                img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
                label = Variable(imgs["label"])

                if self.hparams["gpu_mode"]:    # put variables to gpu
                    img = img.to(self.gpus_list[0])
                    label = label.to(self.gpus_list[0])

                # start train
                for param in self.Net.parameters():
                    param.grad = None

                with torch.cuda.amp.autocast():
                    generated_image, pseudo = self.Net(img)
                    chabonnier_gen = self.criterion(generated_image, label)
                    chabonnier_pseudo = self.criterion(pseudo, label)
                    loss = self.hparams["gen_lambda"] * chabonnier_gen + self.hparams["pseudo_lambda"] * chabonnier_pseudo
        
                if self.hparams["batch_size"] == 1:
                    if i == 1:
                        myutils.save_trainimg(pseudo, epoch, "pseudo_" + datasetname)
                        myutils.save_trainimg(generated_image, epoch, "generated_" + datasetname)

                train_acc = torch.sum(generated_image == label)
                epoch_loss += loss.item()

                if self.hparams["gpu_mode"]:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    loss.backward()
                    self.optimizer.step()

                #compute time and compute efficiency and print information
                process_time = time.time() - start_time
                #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


            if (epoch+1) % (self.hparams["snapshots"]) == 0:
                myutils.checkpointGenerate(epoch, self.hparams, self.Net, datasetname)


            epoch_time = time.time() - epoch_time 
            Accuracy = 100*train_acc / len(dataloader)
            self.writer.add_scalar('loss', epoch_loss, global_step=epoch)
            self.writer.add_scalar('accuracy',Accuracy, global_step=epoch)
            print("===> Epoch {} Complete: Avg. loss: {:.4f} Accuracy {}, Epoch Time: {:.3f} seconds".format(epoch, ((epoch_loss/2) / len(dataloader)), Accuracy, epoch_time))
            print("\n")
            epoch_time = time.time()

def sparse_training(Net, optimizer, dataloader, criterion, params, scaler):
        epoch_loss = 0
        Net.train()

        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            img = Variable(imgs["img"])
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"])

            img = img.cuda()
            label = label.cuda()

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image, pseudo = Net(img)
                chabonnier_gen = criterion(generated_image, label)
                chabonnier_pseudo = criterion(pseudo, label)
                loss = params["gen_lambda"] * chabonnier_gen + params["pseudo_lambda"] * chabonnier_pseudo

            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

def test(Net, dataloader):
    acc = 0
    Net.eval()
    for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar
        img = Variable(imgs["img"])
        img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
        label = Variable(imgs["label"])

        img = img.cuda()
        label = label.cuda()

        with torch.cuda.amp.autocast():
            generated_image, _ = Net(img)

        accuracy = (generated_image == label).sum()

        acc += accuracy

    acc = acc/dataloader.__len__()
    return acc.item()
            
