hparams = {
    "seed": 123,
    "gpus": 1,
    "gpu_mode": True,
    "crop_size": None,
    "resume": False,
    "train_data_path": "C:/Data/dehaze/prepared/",
    "augment_data": False,
    "epochs_o_haze": 100,
    "epochs_nh_haze": 100,
    "epochs_cityscapes": 300,
    "batch_size": 1,
    "crit_lambda": 1,
    "threads": 0,
    "height":128,
    "width":128,
    "lr": 0.0004,
    "beta1": 0.9,
    "beta2": 0.999,
    "mhac_filter": 256,
    "mha_filter": 16,
    "num_mhablock": 8,
    "num_mhac": 10,
    "num_parallel_conv": 2,
    "kernel_list": [3,5,7],
    "pad_list": [4,12,24],
    "save_folder": "./weights/",
    "model_type": "Dehaze",
    "snapshots": 10
}