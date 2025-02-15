from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import os
import ssl
import sys
import uuid
import shutil
import random
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
ssl._create_default_https_context = ssl._create_unverified_context
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection import IntersectionOverUnion
from trainer.base import BaseAgent
from model.Od_Model import get_od_model
from datasets.awd_od_frcnn import AWD_DataLoader_Od_FRCNN
from scheduler.get_lr_scheduler import get_lr_scheduler
from tensorboardX import SummaryWriter
from utils.misc import print_cuda_statistics
from utils.avg_valuemeter import AverageValueMeter    

cudnn.benchmark = True


class FRCNN_Od_Agent(BaseAgent):
    def __init__(self, config,comet_ml):
        super().__init__(config)
        self.comet = comet_ml
        self.model = get_od_model(config=config)
        self.data_loader = AWD_DataLoader_Od_FRCNN(config=config,device=None)

        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr=self.config.learning_rate,
                                        momentum=self.config.momentum, 
                                        weight_decay=self.config.weight_decay,
                                        nesterov=True)
        self.loss__name__ = "loss_score"
        self.scheduler = get_lr_scheduler(config,self.optimizer)


        self.current_epoch = 0
        self.best_metric = 0
        self.met_list = ['map','map_50','map_75','map_small',
                    'map_medium','map_large',
                    'mar_1','mar_10','mar_100',
                    'mar_small','mar_medium','mar_large']
        self.metric_MAP = MeanAveragePrecision(iou_type="bbox",class_metrics=True)

        self.metric_IoU_01 = IntersectionOverUnion(iou_threshold=0.1,class_metrics=True)
        self.metric_IoU_03 = IntersectionOverUnion(iou_threshold=0.3,class_metrics=True)
        self.metric_IoU_05 = IntersectionOverUnion(iou_threshold=0.5,class_metrics=True)

        self.verbose = True
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")
        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")


        if(self.config.checkpoint_file == "False" or self.config.checkpoint_file == "Training_Logs_Location_Placeholder"):
            print("\nNo Checkpoint\n")
        else:
            self.load_checkpoint(self.config.checkpoint_file)
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.config.exp_name)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.3}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s
    
    def load_checkpoint(self, file_name):
        try:
            self.logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.config.checkpoint_dir + file_name)
        if is_best:
            torch.save(state, self.config.checkpoint_dir + file_name)
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self,config):
        try:
            if config.data_mode == "training":
                self.train(config=config)
                self.test(config=config)
            else:
                self.test(config=config)
                
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self,config):
        epoch = -1
        valid_logs = self.validate()
        for metrics_name in valid_logs.keys():
            self.comet.log_metric(metrics_name,valid_logs[metrics_name], epoch=epoch)
        self.best_metric = valid_logs["valmap"]
        print(valid_logs)


        filename = str(epoch)+str(datetime.utcnow().strftime('_%Y_%m_%d__%H_%M_%S_'))+str(valid_logs["valmap"])
        self.save_checkpoint(file_name=filename, is_best=1)
        if self.config.max_epoch > 0 :
            for epoch in range(1, self.config.max_epoch + 1):
                print("----------EPOCH-----",self.current_epoch,"----------")
                train_logs = self.train_one_epoch()
                self.scheduler.step()

                self.comet.log_metric("train_loss", train_logs["loss_score"], epoch=epoch)

                valid_logs = self.validate()
                for metrics_name in valid_logs.keys():
                    self.comet.log_metric(metrics_name,valid_logs[metrics_name], epoch=epoch)
                self.comet.log_metric("lr",self.optimizer.param_groups[0]['lr'], epoch=epoch)
                filename = str(self.current_epoch)+str(datetime.utcnow().strftime('_%Y_%m_%d__%H_%M_%S_'))+str(valid_logs["valmap"])+".pth.tar"
        
                if(valid_logs["valmap"]>self.best_metric):
                    self.best_metric = valid_logs["valmap"]
                    self.save_checkpoint(file_name=filename, is_best=1)
                    
                self.current_epoch += 1
               


    def finalize(self):
        print("experiment done")

    def test(self,config):
        self.logs_test_one_epoch = {}
        self.loss_meter_test_one_epoch = AverageValueMeter()
        self.metrics_meters_test_one_epoch = {}
        
        if config.data_mode == "training":
            del self.model
            self.model = get_od_model(config=config)
            self.load_checkpoint(self.config.checkpoint_dir + 'model_best.pth.tar')
            self.model = self.model.to(self.device)

        self.model.eval()
        with torch.no_grad():
            with tqdm(
                self.data_loader.test_dataloader,
                desc="test",
                file=sys.stdout,
                disable=not (self.verbose),
            ) as iterator:
                for images, targets, image_ids in iterator:
                    batch_len = len(images)
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    preds_list = self.model.forward(images)

                    target_list = []
                    for i in range(batch_len):
                        tar_box = targets[i]['boxes']
                        lab_box = targets[i]['labels']
                        target_list.append(
                            dict(
                                boxes=tar_box,
                                labels=lab_box
                            )
                        )

                    self.metric_MAP.update(preds_list, target_list)
                    self.metric_IoU_01.update(preds_list, target_list)
                    self.metric_IoU_03.update(preds_list, target_list)
                    self.metric_IoU_05.update(preds_list, target_list)

                    if self.verbose:
                        s = self._format_logs(self.logs_test_one_epoch)
                        iterator.set_postfix_str(s)
        
            map_dict = self.metric_MAP.compute()
            for met_name in self.met_list:
                self.logs_test_one_epoch[str('test'+met_name)]=map_dict[met_name].item()

            iou_dict01 = self.metric_IoU_01.compute()
            for iou_keys in iou_dict01.keys():
                self.logs_test_one_epoch[str('test'+iou_keys+'th_0_1')]= iou_dict01[iou_keys].item()
            iou_dict03 = self.metric_IoU_03.compute()
            for iou_keys in iou_dict03.keys():
                self.logs_test_one_epoch[str('test'+iou_keys+'th_0_3')]= iou_dict03[iou_keys].item()
            iou_dict05 = self.metric_IoU_05.compute()
            for iou_keys in iou_dict05.keys():
                self.logs_test_one_epoch[str('test'+iou_keys+'th_0_5')]= iou_dict05[iou_keys].item()
            
            self.metric_MAP.reset()
            self.metric_IoU_01.reset()
            self.metric_IoU_03.reset()
            self.metric_IoU_05.reset()

        test_logs = self.logs_test_one_epoch
        print("Evaluation on Test Data: ")
        for metrics_name in test_logs.keys():
            self.comet.log_metric(metrics_name,test_logs[metrics_name], epoch=self.config.max_epoch)
            print(f"{metrics_name}: {test_logs[metrics_name]:.4f}")

        valid_logs = self.validate()
        for metrics_name in valid_logs.keys():
            self.comet.log_metric(metrics_name,valid_logs[metrics_name], epoch=self.config.max_epoch)
            print(f"{metrics_name}: {valid_logs[metrics_name]:.4f}")
                
        self.best_metric = valid_logs["valmap"]
    
    def train_one_epoch(self):
        self.logs_train_one_epoch = {}
        self.loss_meter_train_one_epoch = AverageValueMeter()
        self.model.train()

        with tqdm(
            self.data_loader.train_loader,
            desc="train",
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for images, targets, image_ids in iterator:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model.forward(images,targets)
                losses = sum(loss for loss in loss_dict.values()) 
                loss_value = losses.cpu().detach().numpy()
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                self.loss_meter_train_one_epoch.add(loss_value)
                loss_logs = {self.loss__name__: self.loss_meter_train_one_epoch.mean}
                self.logs_train_one_epoch.update(loss_logs)

                if self.verbose:
                    s = self._format_logs(self.logs_train_one_epoch)
                    iterator.set_postfix_str(s)

        return self.logs_train_one_epoch

    
    def validate(self):
        self.logs_val_one_epoch = {}
        self.metrics_meters_val_one_epoch = {}

        self.model.eval()
        with torch.no_grad():
            with tqdm(
                self.data_loader.valid_loader,
                desc="validation",
                file=sys.stdout,
                disable=not (self.verbose),
            ) as iterator:
                for images, targets, image_ids in iterator:
                    batch_len = len(images)
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    preds_list = self.model.forward(images)
                    target_list = []
                    for i in range(batch_len):
                        tar_box = targets[i]['boxes']
                        lab_box = targets[i]['labels']
                        target_list.append(
                            dict(
                                boxes=tar_box,
                                labels=lab_box
                            )
                        )
                    self.metric_MAP.update(preds_list, target_list)
                    self.metric_IoU_01.update(preds_list, target_list)
                    self.metric_IoU_03.update(preds_list, target_list)
                    self.metric_IoU_05.update(preds_list, target_list)

            map_dict = self.metric_MAP.compute()
            for met_name in self.met_list:
                self.logs_val_one_epoch[str('val'+met_name)]=map_dict[met_name].item()

            iou_dict01 = self.metric_IoU_01.compute()
            for iou_keys in iou_dict01.keys():
                self.logs_val_one_epoch[str('val'+iou_keys+'th_0_1')]= iou_dict01[iou_keys].item()
            iou_dict03 = self.metric_IoU_03.compute()
            for iou_keys in iou_dict03.keys():
                self.logs_val_one_epoch[str('val'+iou_keys+'th_0_3')]= iou_dict03[iou_keys].item()
            iou_dict05 = self.metric_IoU_05.compute()
            for iou_keys in iou_dict05.keys():
                self.logs_val_one_epoch[str('val'+iou_keys+'th_0_5')]= iou_dict05[iou_keys].item()
            
            self.metric_MAP.reset()
            self.metric_IoU_01.reset()
            self.metric_IoU_03.reset()
            self.metric_IoU_05.reset()

        return self.logs_val_one_epoch

