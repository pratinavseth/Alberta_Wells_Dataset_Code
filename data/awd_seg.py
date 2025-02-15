import h5py
import os
import torch
import pandas as pd
import numpy as np
from einops import rearrange

#Segmentation Map Dataset for Binary Segmentaion , Multi-class segmentaion and Well State Based Segmentation

class AWD_Dataset_Seg(torch.utils.data.Dataset):
    def __init__(self, device,config, hdf5_file, label_type, transform=None,inchannels=4):
        self.csv_file = hdf5_file
        self.label_type = label_type
        self.transform = transform
        df = pd.read_csv(self.csv_file)
        df = df.sample(frac=1).reset_index(drop=True)
        self.image_ids = df['image_id'].tolist()
        self.hdf5_file_paths = df.set_index('image_id')['hdf5_file_path'].to_dict()
        self.num_images = len(self.image_ids)
        self.device = device
        self.class_focus = config.class_focus
        self.inchannels = inchannels

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        hdf5_file_path = self.hdf5_file_paths[image_id]
        image, mask = self.load_and_process(image_id, hdf5_file_path)
        return image, mask

    def __len__(self):
        return len(self.image_ids)

    def load_and_process(self, image_id, hdf5_file_path):
        try:
            with h5py.File(hdf5_file_path, 'r', libver='latest', swmr=True) as hdf5:
                image = hdf5['image'][...]
                labels = hdf5['label'][self.label_type][...]
        except KeyError as e:
            print(f"KeyError: {e}")
            with h5py.File(hdf5_file_path, 'r', libver='latest', swmr=True) as hdf5:
                labels = hdf5['label']['binary_seg_maps'][...]
            if np.array_equal(np.unique(labels), [0]):
                print("error solved")
            else:
                print("ERROR PROB",np.unique(labels))
                return

        image = torch.tensor(image).float()
        image = image / 10000
        if not self.inchannels==4:
            image = image[:self.inchannels]

        if self.label_type in ["binary_seg_maps"]:
            mask = torch.tensor(labels)
            mask[mask > 0] = 1
            mask = torch.nn.functional.one_hot(mask, 2)
            mask = mask[:, :, 1].float().unsqueeze(0)  # Add channel dimension                               
            
        elif self.label_type in ["multi_class_seg_maps"]:
            mask = torch.tensor(labels)
            mask = torch.nn.functional.one_hot(mask, num_classes=4).permute(2, 0, 1).float()
            if not self.class_focus == -1:
                mask = mask[self.class_focus,:,:].unsqueeze(0)
                mask = mask.unsqueeze(0)

        if self.transform:
            image, mask = self.transform(image.unsqueeze(0), mask.unsqueeze(0))
            image = image.squeeze(0)
            mask = mask.squeeze(0)

        return image.float(), mask.float()

    def _adjust_shapes(self, image, mask):
        if image.shape[2] != 4 or mask.shape[0] != 1:
            if image.shape[0] == 4:
                image = image.transpose(2, 0, 1)
            elif image.shape[1] == 4:
                image = image.transpose(0, 2, 1)
                if mask.shape[0] == 1:
                    if image.shape[0] != mask.shape[1]:
                        image = image.transpose(1, 0, 2)
            if mask.shape[0] == 1:
                mask = mask.transpose(2, 0, 1)

        if (image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]):
            raise ValueError(f'The shape of object {image.shape} is incorrect')

        return image, mask