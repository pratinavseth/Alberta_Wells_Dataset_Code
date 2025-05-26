from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import os
import ssl
import sys
import shutil
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.backends import cudnn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection import IntersectionOverUnion
from trainer.base import BaseAgent
from model.Od_Model import get_od_model
from datasets.awd_od_frcnn import AWD_DataLoader_Od_FRCNN
from scheduler.get_lr_scheduler import get_lr_scheduler
from tensorboardX import SummaryWriter
from utils.misc import print_cuda_statistics
from utils.avg_valuemeter import AverageValueMeter    
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup



ssl._create_default_https_context = ssl._create_unverified_context
cudnn.benchmark = True


class DETR_Od_Agent(BaseAgent):
    def __init__(self, config, comet_ml):
        super().__init__(config)
        self.comet = comet_ml
        self.model = get_od_model(config=config)      

        self.data_loader = AWD_DataLoader_Od_FRCNN(config=config, device=None)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        total_training_steps = len(self.data_loader.train_loader) * config.max_epoch
        warmup_steps = int(0.1 * total_training_steps)  # 10% warmup

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )

        self.loss__name__ = "loss_score"

        self.current_epoch = 0
        self.best_metric = 0

        self.metric_MAP = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        self.metric_IoU_01 = IntersectionOverUnion(iou_threshold=0.1, class_metrics=True)
        self.metric_IoU_03 = IntersectionOverUnion(iou_threshold=0.3, class_metrics=True)
        self.metric_IoU_05 = IntersectionOverUnion(iou_threshold=0.5, class_metrics=True)
        self.met_list = list(self.metric_MAP.compute().keys())

        self.verbose = True
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
            torch.cuda.set_device(config.gpu_device)
            print_cuda_statistics()
            self.logger.info("Program will run on *****GPU-CUDA*****")
        else:
            self.logger.info("Program will run on *****CPU*****")

        self.model = self.model.to(self.device)

        if config.checkpoint_file not in ["False", "Training_Logs_Location_Placeholder"]:
            self.load_checkpoint(config.checkpoint_file)
        else:
            self.logger.info("**First time to train**")

        self.summary_writer = SummaryWriter(log_dir=config.summary_dir, comment=config.exp_name)

    def _format_logs(self, logs):
        return ", ".join(f"{k} - {v:.3f}" for k, v in logs.items())

    def _sanitize_boxes(self,boxes):
        x1 = torch.min(boxes[:, 0], boxes[:, 2])
        y1 = torch.min(boxes[:, 1], boxes[:, 3])
        x2 = torch.max(boxes[:, 0], boxes[:, 2])
        y2 = torch.max(boxes[:, 1], boxes[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=-1)


    def _voc_to_coco_boxes(self,boxes, h=256, w=256):
        boxes = self._sanitize_boxes(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes = torch.clamp(boxes, 0.0, 1.0)
        return boxes
    
    def denormalize_xyxy(self, boxes, h=256, w=256):
        """
        Input:  boxes in normalized [x1, y1, x2, y2] format
        Output: boxes in absolute pixel coordinates
        """
        x1, y1, x2, y2 = boxes.unbind(-1)
        x1 = x1 * w
        y1 = y1 * h
        x2 = x2 * w
        y2 = y2 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def load_checkpoint(self, file_name):
        try:
            checkpoint = torch.load(file_name, map_location=self.device)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(f"Checkpoint loaded from {file_name} (epoch {checkpoint['epoch']})")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {file_name}: {e}")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        save_path = os.path.join(self.config.checkpoint_dir, file_name)
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))

    def run(self, config):
        try:
            if config.data_mode == "training":
                self.train(config)
                self.test(config)
            else:
                self.test(config)
        except KeyboardInterrupt:
            self.logger.info("Training interrupted.")

    def _compute_loss(self, images, targets):
        pixel_values = torch.stack(images).to(self.device)
        targets = [{"class_labels": t["labels"].to(self.device), "boxes": t["boxes"].to(self.device)} for t in targets]
        outputs = self.model(pixel_values=pixel_values, labels=targets)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        else:
            return sum(v for v in outputs.values() if isinstance(v, torch.Tensor))


    def train(self, config):
        epoch = -1
        valid_logs = self.validate()
        for k, v in valid_logs.items():
            self.comet.log_metric(k, v, epoch=epoch)
        self.best_metric = valid_logs.get("valmap", 0)
        print(valid_logs)
        self.save_checkpoint(f"{epoch}_{datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S')}.pth.tar", is_best=True)

        for epoch in range(1, config.max_epoch + 1):
            print(f"---------- EPOCH {self.current_epoch} ----------")
            train_logs = self.train_one_epoch()
            self.comet.log_metric("train_loss", train_logs["loss_score"], epoch=epoch)

            valid_logs = self.validate()
            for k, v in valid_logs.items():
                self.comet.log_metric(k, v, epoch=epoch)

            self.comet.log_metric("lr", self.optimizer.param_groups[0]['lr'], epoch=epoch)

            if valid_logs["valmap"] > self.best_metric:
                self.best_metric = valid_logs["valmap"]
                self.save_checkpoint(
                    f"{self.current_epoch}_{datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S')}_{self.best_metric:.4f}.pth.tar",
                    is_best=True)

            self.current_epoch += 1

    def train_one_epoch(self):
        self.logs_train_one_epoch = {}
        self.loss_meter_train_one_epoch = AverageValueMeter()
        self.model.train()

        scaler = GradScaler()
        use_amp = self.current_epoch >= 2 # Use AMP only after warmup


        with tqdm(self.data_loader.train_loader, desc="train", file=sys.stdout, disable=not self.verbose) as iterator:
            for images, targets, image_ids in iterator:
                pixel_values = torch.stack(images).to(self.device)

                targets = [
                    {
                        "class_labels": t["labels"].to(self.device),
                        "boxes": self._voc_to_coco_boxes(t["boxes"].to(self.device))
                    }
                    for t in targets
                ]

                with autocast(enabled=use_amp):
                    outputs = self.model(pixel_values=pixel_values, labels=targets)
                    if hasattr(outputs, "pred_boxes"):
                        outputs.pred_boxes = torch.clamp(outputs.pred_boxes, 0.0, 1.0)
                    loss = outputs.loss
                    loss_value = loss.item()
              
                self.optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.loss_meter_train_one_epoch.add(loss_value)
                self.logs_train_one_epoch[self.loss__name__] = self.loss_meter_train_one_epoch.mean

                if self.verbose:
                    iterator.set_postfix_str(self._format_logs(self.logs_train_one_epoch))

        return self.logs_train_one_epoch

    def validate(self):
        self.model.eval()
        logs_val = {}

        with torch.no_grad():
            for images, targets, _ in tqdm(self.data_loader.valid_loader, desc="validation", file=sys.stdout, disable=not self.verbose):
                images = [img.to(self.device) for img in images]
                targets = [
                    {
                        "class_labels": t["labels"].to(self.device),
                        "boxes": self._voc_to_coco_boxes(t["boxes"].to(self.device))
                    }
                    for t in targets
                ]


                outputs = self.model(pixel_values=torch.stack(images), labels=targets)
                probs = outputs.logits.softmax(-1)[..., :-1]
                scores, pred_labels = probs.max(-1)


                preds_formatted = []
                for i in range(len(outputs.pred_boxes)):
                    preds_formatted.append({
                        "boxes": self.denormalize_xyxy(outputs.pred_boxes[i].detach().cpu()),
                        "scores": scores[i].detach().cpu(),
                        "labels": pred_labels[i].detach().cpu()
                    })

                targets_formatted = [{"boxes": self.denormalize_xyxy(t["boxes"].detach().cpu()), "labels": t["class_labels"].detach().cpu()} for t in targets]

                self.metric_MAP.update(preds_formatted, targets_formatted)
                self.metric_IoU_01.update(preds_formatted, targets_formatted)
                self.metric_IoU_03.update(preds_formatted, targets_formatted)
                self.metric_IoU_05.update(preds_formatted, targets_formatted)

        def safe_log_metrics(prefix, metric_output):
            for k, v in metric_output.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        logs_val[f"{prefix}{k}"] = v.item()
                    else:
                        for i, val in enumerate(v):
                            logs_val[f"{prefix}{k}_class_{i}"] = val.item()
                else:
                    logs_val[f"{prefix}{k}"] = v

        safe_log_metrics("val", self.metric_MAP.compute())
        safe_log_metrics("val_iou_0_1_", self.metric_IoU_01.compute())
        safe_log_metrics("val_iou_0_3_", self.metric_IoU_03.compute())
        safe_log_metrics("val_iou_0_5_", self.metric_IoU_05.compute())

        self.metric_MAP.reset()
        self.metric_IoU_01.reset()
        self.metric_IoU_03.reset()
        self.metric_IoU_05.reset()
        return logs_val

    def test(self, config):
        if config.data_mode == "training":
            del self.model
            self.model = get_od_model(config=config).to(self.device)
            self.load_checkpoint(os.path.join(config.checkpoint_dir, "model_best.pth.tar"))

        self.model.eval()
        logs_test = {}

        with torch.no_grad():
            for images, targets, _ in tqdm(self.data_loader.test_dataloader, desc="test", file=sys.stdout, disable=not self.verbose):
                images = [img.to(self.device) for img in images]
                targets = [
                    {
                        "class_labels": t["labels"].to(self.device),
                        "boxes": self._voc_to_coco_boxes(t["boxes"].to(self.device))
                    }
                    for t in targets
                ]
                outputs = self.model(pixel_values=torch.stack(images), labels=targets)

                probs = outputs.logits.softmax(-1)[..., :-1]
                scores, pred_labels = probs.max(-1)

                preds_formatted = []
                for i in range(len(outputs.pred_boxes)):
                    preds_formatted.append({
                        "boxes": self.denormalize_xyxy(outputs.pred_boxes[i].detach().cpu()),
                        "scores": scores[i].detach().cpu(),
                        "labels": pred_labels[i].detach().cpu()
                    })

                targets_formatted = [{"boxes": self.denormalize_xyxy(t["boxes"].detach().cpu()), "labels": t["class_labels"].detach().cpu()} for t in targets]

                self.metric_MAP.update(preds_formatted, targets_formatted)
                self.metric_IoU_01.update(preds_formatted, targets_formatted)
                self.metric_IoU_03.update(preds_formatted, targets_formatted)
                self.metric_IoU_05.update(preds_formatted, targets_formatted)

        def safe_log_metrics(prefix, metric_output):
            for k, v in metric_output.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        logs_test[f"{prefix}{k}"] = v.item()
                    else:
                        for i, val in enumerate(v):
                            logs_test[f"{prefix}{k}_class_{i}"] = val.item()
                else:
                    logs_test[f"{prefix}{k}"] = v

        safe_log_metrics("test", self.metric_MAP.compute())
        safe_log_metrics("test_iou_0_1_", self.metric_IoU_01.compute())
        safe_log_metrics("test_iou_0_3_", self.metric_IoU_03.compute())
        safe_log_metrics("test_iou_0_5_", self.metric_IoU_05.compute())

        self.metric_MAP.reset()
        self.metric_IoU_01.reset()
        self.metric_IoU_03.reset()
        self.metric_IoU_05.reset()

        for k, v in logs_test.items():
            self.comet.log_metric(k, v, epoch=config.max_epoch)
            print(f"{k}: {v:.4f}")
    
    def finalize(self):
        print("experiment done")
