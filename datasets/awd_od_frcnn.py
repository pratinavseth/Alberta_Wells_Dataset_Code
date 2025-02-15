import os
import torch
import logging
import time
from torch.utils.data import DataLoader
#Importing  Faster RCNN Object Detection Dataset Class
from data.awd_od_frcnn import AWD_Dataset_Od_FRCNN
#Importing Object Detection Augmentation Function
from utils.adw_od_aug import object_detection_train_augmentation, object_detection_eval_augmentation

class AWD_DataLoader_Od_FRCNN:
    def __init__(self, config, device):
        self.train_hdf5_file = config.train_hdf5_file
        self.eval_hdf5_file = config.eval_hdf5_file
        self.test_hdf5_file = config.test_hdf5_file
        self.train_bs = config.bs_train
        self.evalt_bs = config.bs_evalt
        self.train_workers = config.train_workers
        self.inchannels=config.in_channels
        self.evalt_workers = config.evalt_workers
        self.device = device

        if config.data_mode == "training":
            start_time = time.time()
            eval_ds = AWD_Dataset_Od_FRCNN(eval=True,
                                            device=self.device, 
                                            config=config, 
                                            hdf5_file=self.eval_hdf5_file,  
                                            transform=object_detection_eval_augmentation(config), inchannels=self.inchannels)
            print("Time Taken for eval dataset initialization: ", time.time() - start_time)
            
            start_time = time.time()
            test_ds = AWD_Dataset_Od_FRCNN(eval=True,
                                            device=self.device, 
                                            config=config, 
                                            hdf5_file=self.test_hdf5_file, 
                                            transform=object_detection_eval_augmentation(config), inchannels=self.inchannels)
            print("Time Taken for test set initialization: ", time.time() - start_time)
            
            start_time = time.time()
            train_ds = AWD_Dataset_Od_FRCNN(eval=False,
                                            device=self.device, 
                                            config=config, 
                                            hdf5_file=self.train_hdf5_file, 
                                            transform=object_detection_train_augmentation(config), inchannels=self.inchannels)
            print("Time Taken for train set initialization: ", time.time() - start_time)
            
            self.train_loader = DataLoader(train_ds, batch_size=self.train_bs, 
                                                shuffle=True, num_workers=self.train_workers,
                                                pin_memory=True, collate_fn=AWD_Dataset_Od_FRCNN.collate_fn)
            self.valid_loader = DataLoader(eval_ds, batch_size=self.evalt_bs, 
                                                shuffle=False, num_workers=self.evalt_workers, 
                                                pin_memory=True, collate_fn=AWD_Dataset_Od_FRCNN.collate_fn)
            self.test_dataloader = DataLoader(test_ds, batch_size=self.evalt_bs, 
                                                shuffle=False, num_workers=self.evalt_workers, 
                                                pin_memory=True, collate_fn=AWD_Dataset_Od_FRCNN.collate_fn)

        elif config.data_mode == "inference_valid_test":
            eval_ds = AWD_Dataset_Od_FRCNN(eval=True,device=self.device, config=config, hdf5_file=self.eval_hdf5_file, transform=object_detection_eval_augmentation(config), inchannels=self.inchannels)
            test_ds = AWD_Dataset_Od_FRCNN(eval=True,device=self.device, config=config, hdf5_file=self.test_hdf5_file, transform=object_detection_eval_augmentation(config), inchannels=self.inchannels)
            self.valid_loader = DataLoader(eval_ds, batch_size=self.evalt_bs, shuffle=False, num_workers=self.evalt_workers, collate_fn=AWD_Dataset_Od_FRCNN.collate_fn)
            self.test_dataloader = DataLoader(test_ds, batch_size=self.evalt_bs, shuffle=False, num_workers=self.evalt_workers, collate_fn=AWD_Dataset_Od_FRCNN.collate_fn)

        else:
            raise Exception("Please specify a valid mode in data_mode")

    def finalize(self):
        pass