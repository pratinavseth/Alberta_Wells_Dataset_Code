import os
import imageio
import torch
import logging
import time
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
from data.awd_seg import AWD_Dataset_Seg
from utils.adw_bseg_aug import bseg_train_augmentation,bseg_eval_augmentation


class AWD_DataLoader_BSeg:
    def __init__(self, config,device):
        self.train_hdf5_file = config.train_hdf5_file
        self.eval_hdf5_file = config.eval_hdf5_file
        self.test_hdf5_file = config.test_hdf5_file
        self.label_type = config.label_type
        self.train_bs = config.bs_train
        self.evalt_bs = config.bs_evalt
        self.train_workers = config.train_workers
        self.evalt_workers = config.evalt_workers
        self.inchannels = config.in_channels
        self.device = device


        if config.data_mode == "training":
            start_time = time.time()
            eval_ds = AWD_Dataset_Seg(device=self.device,config=config,hdf5_file=self.eval_hdf5_file, label_type=self.label_type, transform=bseg_eval_augmentation(config=config), inchannels=self.inchannels)
            constructor_time = time.time() - start_time
            print("Evaluation Set Time Taken for initialization: ",constructor_time)
            start_time = time.time()
            test_ds = AWD_Dataset_Seg(device=self.device,config=config,hdf5_file=self.test_hdf5_file, label_type=self.label_type, transform=bseg_eval_augmentation(config=config), inchannels=self.inchannels)
            constructor_time = time.time() - start_time
            print("Test Set Time Taken for initialization: ",constructor_time)
            start_time = time.time()
            train_ds = AWD_Dataset_Seg(device=self.device,config=config,hdf5_file=self.train_hdf5_file, label_type=self.label_type, transform=bseg_train_augmentation(config=config), inchannels=self.inchannels)
            constructor_time = time.time() - start_time
            print("Train Set Time Taken for initialization: ",constructor_time)
            self.train_loader = DataLoader(train_ds, batch_size=self.train_bs,
                shuffle=True, num_workers=self.train_workers, pin_memory=True,
                prefetch_factor=self.train_workers,persistent_workers=False)
            self.valid_loader = DataLoader(eval_ds, batch_size=self.evalt_bs, shuffle=False, 
                num_workers=self.evalt_workers, pin_memory=True,
                prefetch_factor=self.evalt_workers,persistent_workers=False)
            self.test_dataloader = DataLoader(test_ds, batch_size=self.evalt_bs, 
                shuffle=False, num_workers=self.evalt_workers, pin_memory=True,
                prefetch_factor=self.evalt_workers,persistent_workers=False)


        elif config.data_mode == "inference_valid_test":
            start_time = time.time()
            eval_ds = AWD_Dataset_Seg(device=self.device,config=config,hdf5_file=self.eval_hdf5_file, label_type=self.label_type, transform=bseg_eval_augmentation(config=config), inchannels=self.inchannels)
            constructor_time = time.time() - start_time
            print("Evaluation Set Time Taken for initialization: ",constructor_time)
            start_time = time.time()
            test_ds = AWD_Dataset_Seg(device=self.device,config=config,hdf5_file=self.test_hdf5_file, label_type=self.label_type, transform=bseg_eval_augmentation(config=config), inchannels=self.inchannels)
            constructor_time = time.time() - start_time
            print("Test Set Time Taken for initialization: ",constructor_time)
            self.valid_loader = DataLoader(eval_ds, batch_size=self.evalt_bs, shuffle=False, 
                num_workers=self.evalt_workers, pin_memory=True,
                prefetch_factor=self.evalt_workers,persistent_workers=False)
            self.test_dataloader = DataLoader(test_ds, batch_size=self.evalt_bs, 
                shuffle=False, num_workers=self.evalt_workers, pin_memory=True,
                prefetch_factor=self.evalt_workers,persistent_workers=False)

        else:
            raise Exception("Please specify a valid mode in data_mode")

    def finalize(self):
        pass

