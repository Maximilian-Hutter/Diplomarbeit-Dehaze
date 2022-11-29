hparams = {
    "seed": 123,
    "gpus": 1,
    "gpu_mode": True,
    "crop_size": None,
    "resume": False,
    "train_data_path": "C:/Data/dehaze/prepared/",
    "augment_data": False,
    "epochs_o_haze": 150,
    "epochs_nh_haze": 150,
    "epochs_cityscapes": 500,
    "batch_size": 1,
    "gen_lambda": 0.8,
    "pseudo_lambda": 0.3,
    "threads": 0,
    "height":512, #1280, 512, 288 niedrigste zahl = 248
    "width":288,    #720, 288, 288 solange durch 8 teilbar & >= 248
    "lr": 8.214e-06,
    "beta1": 0.9595,
    "beta2": 0.9901,
    "mhac_filter": 32,  # 256
    "mha_filter": 8,    #16
    "num_mhablock": 6,  # 8
    "num_mhac": 7, # 10
    "num_parallel_conv": 2,
    "kernel_list": [3,5,7],
    "pad_list": [4,12,24],
    "start_epoch": 0,
    "save_folder": "./weights/",
    "model_type": "Dehaze",
    "scale_factor": 1,
    "snapshots": 1
}