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
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Accuracy, Recall, Precision
from trainer.base import BaseAgent
from metrics.BACC import BACC
from model.Seg_Model import get_seg_model
from losses.get_lossfn import get_lossfn
from datasets.awd_bseg import AWD_DataLoader_BSeg
from datasets.awd_mseg import AWD_DataLoader_MSeg
from scheduler.get_lr_scheduler import get_lr_scheduler
from tensorboardX import SummaryWriter
from utils.misc import print_cuda_statistics
from utils.avg_valuemeter import AverageValueMeter

cudnn.benchmark = True


class Seg_Agent_LRS(BaseAgent):

    def __init__(self, config,comet_ml):
        super().__init__(config)
        self.comet = comet_ml

        self.model = get_seg_model(config=config)
        if config['data_loader'] == "AWD_BSeg":
            print("Binary Segmentation Dataloader")
            self.data_loader = AWD_DataLoader_BSeg(config=config,device=None)
        elif config['data_loader'] == "AWD_MSeg":
            print("Multi-Class Segmentation Dataloader")
            self.data_loader = AWD_DataLoader_MSeg(config=config)
        else:
            print("Other Dataloader")
            self.data_loader = AWD_DataLoader_BSeg(config=config)

        self.loss = get_lossfn(config=config)
        self.loss.__name__ = "loss_score"
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = get_lr_scheduler(config,self.optimizer)
        self.current_epoch = 0
        self.best_metric = 0
        self.class_focus = config.class_focus
        self.threshold = config.thresholding
        self.metrics_list = [
            smp.utils.metrics.IoU(threshold=self.threshold),
            smp.utils.metrics.Fscore(threshold=self.threshold),
            BACC(threshold=self.threshold),
            smp.utils.metrics.Accuracy(threshold=self.threshold),
            smp.utils.metrics.Recall(threshold=self.threshold),
            smp.utils.metrics.Precision(threshold=self.threshold)]
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
            self.loss = self.loss.to(self.device)
            for metric in self.metrics_list:
                metric.to(self.device)
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
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {})\n".format(self.config.checkpoint_dir, checkpoint['epoch']))
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
        print("train")
        valid_logs = self.validate()
        self.comet.log_metric("val_accuracy", valid_logs["accuracy"], epoch=epoch)
        self.comet.log_metric("val_loss", valid_logs["loss_score"], epoch=epoch)
        self.comet.log_metric("val_fscore", valid_logs["fscore"], epoch=epoch) 
        self.comet.log_metric("val_iou", valid_logs["iou_score"], epoch=epoch)
        self.comet.log_metric("val_precision", valid_logs["precision"], epoch=epoch) 
        self.comet.log_metric("val_recall", valid_logs["recall"], epoch=epoch)
        self.comet.log_metric("val_BACC_Custom", valid_logs["BACC_Custom"], epoch=epoch)
        self.comet.log_metric("lr",self.optimizer.param_groups[0]['lr'], epoch=epoch)
        self.best_metric = valid_logs["iou_score"]
        filename = str(epoch)+"_"+str(self.config.seed)+"_"+str(datetime.utcnow().strftime('_%Y_%m_%d__%H_%M_%S_'))+str(valid_logs["iou_score"])
        self.save_checkpoint(file_name=filename, is_best=1)
        
        if self.config.max_epoch > 0 :
            for epoch in range(1, self.config.max_epoch + 1):
                print("----------EPOCH-----",self.current_epoch,"----------")
                train_logs = self.train_one_epoch()
                self.scheduler.step()
                self.comet.log_metric("train_accuracy", train_logs["accuracy"], epoch=epoch) # TODO: one line
                self.comet.log_metric("train_loss", train_logs["loss_score"], epoch=epoch)
                self.comet.log_metric("train_fscore", train_logs["fscore"], epoch=epoch)
                self.comet.log_metric("train_iou", train_logs["iou_score"], epoch=epoch)
                self.comet.log_metric("train_precision", train_logs["precision"], epoch=epoch)
                self.comet.log_metric("train_recall", train_logs["recall"], epoch=epoch)
                self.comet.log_metric("train_BACC_Custom", train_logs["BACC_Custom"], epoch=epoch)

                valid_logs = self.validate()
                self.comet.log_metric("val_accuracy", valid_logs["accuracy"], epoch=epoch)
                self.comet.log_metric("val_loss", valid_logs["loss_score"], epoch=epoch)
                self.comet.log_metric("val_fscore", valid_logs["fscore"], epoch=epoch) 
                self.comet.log_metric("val_iou", valid_logs["iou_score"], epoch=epoch)
                self.comet.log_metric("val_precision", valid_logs["precision"], epoch=epoch) 
                self.comet.log_metric("val_recall", valid_logs["recall"], epoch=epoch)
                self.comet.log_metric("val_BACC_Custom", valid_logs["BACC_Custom"], epoch=epoch)
                self.comet.log_metric("lr",self.scheduler.get_last_lr(), epoch=epoch)
                
                filename = str(self.current_epoch)+"_"+str(self.config.seed)+"_"+str(datetime.utcnow().strftime('_%Y_%m_%d__%H_%M_%S_'))+str(valid_logs["iou_score"])+".pth.tar"
                    
                if(valid_logs["iou_score"]>self.best_metric):
                    self.best_metric = valid_logs["iou_score"]
                    self.save_checkpoint(file_name=filename, is_best=1)
                    
                self.current_epoch += 1
               
    def finalize(self):
        print("experiment done")

    def test(self,config):
        self.logs_test_one_epoch = {}
        self.loss_meter_test_one_epoch = AverageValueMeter()
        self.metrics_meters_test_one_epoch = {metric.__name__: AverageValueMeter() for metric in self.metrics_list}
        
        if config.data_mode == "training":
            del self.model
            self.model = get_seg_model(config=config)
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
                for data, target in iterator:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model.forward(data)

                    if self.class_focus == 1:
                        binary_pred = output.to(torch.float32)
                        th_label = torch.argmax(target,dim=1)>0
                        th_label = th_label.int()
                        binary_pred = (binary_pred > self.threshold).float()
                        mask_non_focus_classes_remove = ~((target[:, 2, :, :] == 1) | (target[:, 3, :, :] == 1)) # Exclude no interested classes
                        valid_pred = torch.masked_select(binary_pred, mask_non_focus_classes_remove.bool()) # Keep valid prediction pixels (foreground + background)
                        valid_labels = torch.masked_select(th_label, mask_non_focus_classes_remove.bool()) # Keep valid label pixels (foreground + background)
                        for metric_fn in self.metrics_list:
                            metric_value = metric_fn(valid_pred, valid_labels).cpu().detach().numpy()
                            self.metrics_meters_test_one_epoch[metric_fn.__name__].add(metric_value)
                        metrics_logs = {k: v.mean for k, v in self.metrics_meters_test_one_epoch.items()}
                        self.logs_test_one_epoch.update(metrics_logs)
                        if self.verbose:
                            s = self._format_logs(self.logs_test_one_epoch)
                            iterator.set_postfix_str(s)

                    elif self.class_focus == 2:
                        binary_pred = output.to(torch.float32)
                        th_label = torch.argmax(target,dim=1)>0
                        th_label = th_label.int()
                        binary_pred = (binary_pred > self.threshold).float()
                        mask_non_focus_classes_remove = ~((target[:, 1, :, :] == 1) | (target[:, 3, :, :] == 1))
                        valid_pred = torch.masked_select(binary_pred, mask_non_focus_classes_remove.bool())  # Keep valid prediction pixels (foreground + background)
                        valid_labels = torch.masked_select(th_label, mask_non_focus_classes_remove.bool())  # Keep valid label pixels (foreground + background)
                        for metric_fn in self.metrics_list:
                            metric_value = metric_fn(valid_pred, valid_labels).cpu().detach().numpy()
                            self.metrics_meters_test_one_epoch[metric_fn.__name__].add(metric_value)
                        metrics_logs = {k: v.mean for k, v in self.metrics_meters_test_one_epoch.items()}
                        self.logs_test_one_epoch.update(metrics_logs)
                        if self.verbose:
                            s = self._format_logs(self.logs_test_one_epoch)
                            iterator.set_postfix_str(s)

                    elif self.class_focus == 3:
                        binary_pred = output.to(torch.float32)
                        th_label = torch.argmax(target,dim=1)>0
                        th_label = th_label.int()
                        binary_pred = (binary_pred > self.threshold).float()
                        mask_non_focus_classes_remove = ~((target[:, 2, :, :] == 1) | (target[:, 1, :, :] == 1))
                        valid_pred = torch.masked_select(binary_pred, mask_non_focus_classes_remove.bool())  # Keep valid prediction pixels (foreground + background)
                        valid_labels = torch.masked_select(th_label, mask_non_focus_classes_remove.bool())  # Keep valid label pixels (foreground + background)
                        for metric_fn in self.metrics_list:
                            metric_value = metric_fn(valid_pred, valid_labels).cpu().detach().numpy()
                            self.metrics_meters_test_one_epoch[metric_fn.__name__].add(metric_value)
                        metrics_logs = {k: v.mean for k, v in self.metrics_meters_test_one_epoch.items()}
                        self.logs_test_one_epoch.update(metrics_logs)
                        if self.verbose:
                            s = self._format_logs(self.logs_test_one_epoch)
                            iterator.set_postfix_str(s)
                    else:
                        loss = self.loss(output, target)
                        loss_value = loss.cpu().detach().numpy()
                        self.loss_meter_test_one_epoch.add(loss_value)
                        loss_logs = {self.loss.__name__: self.loss_meter_test_one_epoch.mean}
                        self.logs_test_one_epoch.update(loss_logs)
                        
                        for metric_fn in self.metrics_list:
                            metric_value = metric_fn(output, target).cpu().detach().numpy()
                            self.metrics_meters_test_one_epoch[metric_fn.__name__].add(metric_value)
                        metrics_logs = {k: v.mean for k, v in self.metrics_meters_test_one_epoch.items()}
                        self.logs_test_one_epoch.update(metrics_logs)

                        if self.verbose:
                            s = self._format_logs(self.logs_test_one_epoch)
                            iterator.set_postfix_str(s)
        
        test_logs = self.logs_test_one_epoch
        self.comet.log_metric("test_accuracy", test_logs["accuracy"], epoch = self.config.max_epoch) #TODO: turn into 1 line
        if self.class_focus == 1 or self.class_focus == 2 or self.class_focus == 3:
            pass
        else:
            self.comet.log_metric("test_loss", test_logs["loss_score"], epoch = self.config.max_epoch)
        self.comet.log_metric("test_BACC_Custom", test_logs["BACC_Custom"], epoch=self.config.max_epoch)
        self.comet.log_metric("test_fscore", test_logs["fscore"], epoch = self.config.max_epoch) 
        self.comet.log_metric("test_iou", test_logs["iou_score"], epoch = self.config.max_epoch)
        self.comet.log_metric("test_precision", test_logs["precision"], epoch = self.config.max_epoch) 
        self.comet.log_metric("test_recall", test_logs["recall"], epoch = self.config.max_epoch) 

        print("Evaluation on Test Data: ")
        print(f"Mean IoU Score: {test_logs['iou_score']:.4f}")
        print(f"Mean F Score: {test_logs['fscore']:.4f}")
        print(f"Mean Accuracy : {test_logs['accuracy']:.4f}")
        print(f"Mean Recall : {test_logs['recall']:.4f}")
        print(f"Mean Precision: {test_logs['precision']:.4f}")
        print(f"Mean BACC : {test_logs['BACC_Custom']:.4f}")
    
    def train_one_epoch(self):
        self.logs_train_one_epoch = {}
        self.loss_meter_train_one_epoch = AverageValueMeter()
        self.metrics_meters_train_one_epoch = {metric.__name__: AverageValueMeter() for metric in self.metrics_list}

        self.model.train()
        with tqdm(
            self.data_loader.train_loader,
            desc="train",
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for data, target in iterator:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                loss_value = loss.cpu().detach().numpy()
                self.loss_meter_train_one_epoch.add(loss_value)
                loss_logs = {self.loss.__name__: self.loss_meter_train_one_epoch.mean}
                self.logs_train_one_epoch.update(loss_logs)

                for metric_fn in self.metrics_list:
                    metric_value = metric_fn(output, target).cpu().detach().numpy()
                    self.metrics_meters_train_one_epoch[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in self.metrics_meters_train_one_epoch.items()}
                self.logs_train_one_epoch.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(self.logs_train_one_epoch)
                    iterator.set_postfix_str(s)
        return self.logs_train_one_epoch  
    
    def validate(self):
        self.logs_val_one_epoch = {}
        self.loss_meter_val_one_epoch = AverageValueMeter()
        self.metrics_meters_val_one_epoch = {metric.__name__: AverageValueMeter() for metric in self.metrics_list}
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                self.data_loader.valid_loader,
                desc="validation",
                file=sys.stdout,
                disable=not (self.verbose),
            ) as iterator:
                for data, target in iterator:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model.forward(data)
                    loss = self.loss(output, target)
                    loss_value = loss.cpu().detach().numpy()
                    self.loss_meter_val_one_epoch.add(loss_value)
                    loss_logs = {self.loss.__name__: self.loss_meter_val_one_epoch.mean}
                    self.logs_val_one_epoch.update(loss_logs)
                    for metric_fn in self.metrics_list:
                        metric_value = metric_fn(output, target).cpu().detach().numpy()
                        self.metrics_meters_val_one_epoch[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in self.metrics_meters_val_one_epoch.items()}
                    self.logs_val_one_epoch.update(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(self.logs_val_one_epoch)
                        iterator.set_postfix_str(s)
        
        return self.logs_val_one_epoch