import optuna
import torch
from models import Dehaze
import torch.optim as optim
from training import sparse_training, test
from data import ImageDataset
from loss import ChabonnierLoss
from torch.utils.data import DataLoader
from params import hparams
import numpy as np
import myutils

def objective(trial):
    size = (hparams["height"], hparams["width"])
    scaler = torch.cuda.amp.GradScaler()
    params = {
              "mhac_filter":trial.suggest_categorical("mhac_filer", [64,128,256,512]),
              "mha_filter":trial.suggest_categorical("mha_filer", [8,16,32]),
              "num_mhablock":trial.suggest_int("num_mhablock", 3,10),
              "num_mhac":trial.suggest_int("num_mhac", 3, 12),
              "lr":trial.suggest_loguniform('learning_rate', 1e-6, 1e-1),
              "beta1":trial.suggest_float("beta1", 0.85, 1),
              "beta2":trial.suggest_float("beta2", 0.9, 1),
              }
    
    model = Dehaze(params["mhac_filter"], params["mha_filter"], params["num_mhablock"], params["num_mhac"]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], betas=(params["beta1"],params["beta2"]))
    criterion = ChabonnierLoss(eps = 1e-6).cuda()
    dataloader = DataLoader(ImageDataset(hparams["train_data_path"] + "O-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    testloader = DataLoader(ImageDataset("C:/Data/dehaze/test",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    

    for i in range(20):    
        sparse_training(model, optimizer, dataloader, criterion, hparams, scaler)


    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2

    torch.save(model.state_dict(), "./tune_saves/" + str(round(size_all_mb))+".pth")

    accuracy = test(model, testloader)
    return accuracy

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))