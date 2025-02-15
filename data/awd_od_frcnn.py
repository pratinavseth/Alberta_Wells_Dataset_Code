import h5py
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from albumentations.core.bbox_utils import calculate_bbox_area,normalize_bboxes

#Object Detection Dataset in PASCAL VOC Format for RetinaNet and Faster RCNN

class AWD_Dataset_Od_FRCNN(Dataset):
    def __init__(self, eval, device, config, hdf5_file, transform=None,inchannels=4):
        self.csv_file = hdf5_file
        self.transform_bb,self.transform = transform
        self.eval = eval
        df = pd.read_csv(self.csv_file)
        df = df.sample(frac=1).reset_index(drop=True)
        self.image_ids = df['image_id'].tolist()
        self.hdf5_file_paths = df.set_index('image_id')['hdf5_file_path'].to_dict()
        self.num_images = len(self.image_ids)
        self.device = device
        self.no_classes = config.no_classes
        self.inchannels = inchannels

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        hdf5_file_path = self.hdf5_file_paths[image_id]
        image, bboxes, labels = self.load_and_process(image_id, hdf5_file_path)
        if labels[0] == 0:
            transformed = self.transform(image=image)
            image = transformed['image']
            _,h,w = image.shape
            target = {}
            target['boxes'] = torch.as_tensor(bboxes,dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels,dtype=torch.long)
            target['image_id'] = torch.tensor([index])
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["iscrowd"] = torch.ones((len(labels),), dtype=torch.int64)
        else:
            transformed = self.transform_bb(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
            _,h,w = image.shape
            target = {}
            target['boxes'] = torch.as_tensor(bboxes,dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels,dtype=torch.long)
            target['image_id'] = torch.tensor([index])
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        return image, target, image_id


    def __len__(self):
        return len(self.image_ids)

    def load_and_process(self, image_id, hdf5_file_path):
        with h5py.File(hdf5_file_path, 'r', libver='latest', swmr=True) as hdf5:
            image = hdf5['image'][...]
            label_data = hdf5['label']['bounding_box_annotations'][...]
            if label_data.ndim == 0:
                label_data_str = label_data.item()
            else:
                label_data_str = label_data.decode('utf-8') if isinstance(label_data, bytes) else label_data
            
            annotations = json.loads(label_data_str)

        image = image / 10000.0
        image = image[:self.inchannels]
        image = image.transpose(1,2,0)

        bboxes = []
        labels = []
        areas = []
        for annotation in annotations:
            bbox = annotation['bbox']
            if annotation['category_id'] != 0 and self.no_classes == 1:
                category_id = 1
            else:
                category_id = annotation['category_id']
            bboxes.append((bbox[0],bbox[1],bbox[2]+bbox[0],bbox[3]+bbox[1]))
            labels.append(category_id)

        return image, bboxes, labels

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

