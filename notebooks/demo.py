import os, json, h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from utils.config import process_config
from model.Od_Model import get_od_model
from model.Seg_Model import get_seg_model
from utils.adw_bseg_aug import bseg_eval_augmentation
from utils.adw_od_aug import object_detection_eval_augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os

def denormalize_xyxy(boxes, h=256, w=256):
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

def _sanitize_boxes(boxes):
    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _voc_to_coco_boxes(boxes, h=256, w=256):
    boxes = _sanitize_boxes(boxes)
    coco_boxes = boxes.clone()  # Copy boxes to avoid modification of original boxes
    coco_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = x_max - x_min
    coco_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = y_max - y_min
    coco_boxes = coco_boxes / torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    coco_boxes = torch.clamp(coco_boxes, 0.0, 1.0)
    return coco_boxes

def generate_composite_panel(image_tensor, preds_formatted, targets_formatted, outdir, filename="test7046_comparison_panel.png"):
    image = 2.5 * image_tensor
    print(image_tensor.shape)
    print(image.shape)
    fig, axs = plt.subplots(1, 3, figsize=(30, 20))
    axs[0].imshow(image)
    for bbox, score, label in zip(preds_formatted[0]["boxes"], preds_formatted[0]["scores"], preds_formatted[0]["labels"]):
        if score >= 0.3:
            x1, y1, x2, y2 = bbox.tolist()
            axs[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='blue', linewidth=2))
            axs[0].text(x1, y1 + 30, f"Prob:{score:.2f}", color='blue', fontsize=12)
    axs[0].set_title("Bounding Box Prediction Threshold ≥ 0.3", fontsize=18, weight='bold')
    axs[0].axis("off")
    axs[1].imshow(image)
    for bbox, score, label in zip(preds_formatted[0]["boxes"], preds_formatted[0]["scores"], preds_formatted[0]["labels"]):
        if score >= 0.5:
            x1, y1, x2, y2 = bbox.tolist()
            axs[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            axs[1].text(x1, y1 + 30, f"Prob:{score:.2f}", color='red', fontsize=12)
    axs[1].set_title("Bounding Box Prediction Threshold ≥ 0.5", fontsize=18, weight='bold')
    axs[1].axis("off")
    axs[2].imshow(image)
    for bbox, label in zip(targets_formatted["boxes"], targets_formatted["labels"]):
        x1, y1, x2, y2 = bbox.tolist()
        axs[2].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, edgecolor='white', linewidth=2,alpha=0.3,facecolor='white'))
        axs[2].text(x1, y1 + 30, f"Well Location", color='white', fontsize=12)
    axs[2].set_title("Bounding Box Annotation", fontsize=18, weight='bold')
    axs[2].axis("off")
    plt.tight_layout()
    save_path = os.path.join(outdir, filename)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # optional: reduce inter-panel space
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[INFO] Comparison panel saved to: {save_path}")

# ---------- SETUP ----------
TASK = "SEG"  # or "OD"
H5_PATH = "placeholder"
#H5_PATH = "downloads/adw-files-split/test/7046.h5"
if TASK == "SEG":
    CONFIG_PATH = "placeholder"  # or DETR.json
    CKPT_PATH = "placeholder"
if TASK == "OD":
    CONFIG_PATH = "placeholder"  # or DETR.json
    CKPT_PATH = "placeholder"

OUTDIR = "awd_outputs"
os.makedirs(OUTDIR, exist_ok=True)

config = process_config(type('args', (), {"config": CONFIG_PATH, "CUR_DIR": os.getcwd(), "SEED": 42})())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD AND PREPROCESS IMAGE ----------

# ---------- SEGMENTATION ----------
if TASK == "SEG":
    with h5py.File(H5_PATH, 'r', libver='latest', swmr=True) as hdf5:
        image_np = hdf5['image'][...]
        labels = hdf5['label'][config.label_type][...]
        image = torch.tensor(image_np).float()
        image = image / 10000
        mask = torch.tensor(labels)
        mask[mask > 0] = 1
        mask = torch.nn.functional.one_hot(mask, 2)
        mask = mask[:, :, 1].float().unsqueeze(0)  # Add channel dimension       
        transform = bseg_eval_augmentation(config=config)
        image_tensor, mask = transform(image.unsqueeze(0), mask.unsqueeze(0))
        print(image_tensor.shape,mask.shape)

    model = get_seg_model(config).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device)["state_dict"])
    model.eval()

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        pred = model(image_tensor).numpy().squeeze(0).squeeze(0)
        pred = pred > 0

    vis_img = image_tensor.squeeze(0).numpy()
    print(vis_img.shape)
    vis_img = 2.5 * vis_img.transpose(1, 2, 0)
    fig, axs = plt.subplots(1, 3, figsize=(30, 20))
    axs[0].imshow(vis_img) #hwc required
    axs[0].set_title('Image Sample', fontsize=18, weight='bold')
    axs[0].axis("off")
    axs[1].imshow(pred) #hwc required
    axs[1].set_title('Well Segmentation Prediction', fontsize=18, weight='bold')
    axs[1].axis("off")
    axs[2].imshow(mask.squeeze(0).squeeze(0)) #hwc required
    axs[2].set_title('Well Segmentation Label', fontsize=18, weight='bold')
    axs[2].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # optional: reduce inter-panel space
    fig.savefig(os.path.join(OUTDIR, "test7046_single_segmentation.png"),bbox_inches='tight', pad_inches=0.1)
    fig.show()
    print("saved",os.path.join(OUTDIR, "test7046_single_segmentation.png"))

# ---------- OBJECT DETECTION ----------
elif TASK == "OD":
    transform_bb,transform= object_detection_eval_augmentation(config)
    transform_2 = A.Compose([
        A.Resize(height=256, width=256),
    ], bbox_params=A.BboxParams(format=str(config.bbox_format), min_visibility=0.1, label_fields=['labels'],clip=True))
    transform_3 = A.Compose([
        A.Resize(height=256, width=256),
    ])

    with h5py.File(H5_PATH, 'r', libver='latest', swmr=True) as hdf5:
        image_np = hdf5['image'][...]
        label_data = hdf5['label']['bounding_box_annotations'][...]
        if label_data.ndim == 0:
            label_data_str = label_data.item()
        else:
            label_data_str = label_data.decode('utf-8') if isinstance(label_data, bytes) else label_data
        
        annotations = json.loads(label_data_str)
        image = image_np.transpose(1,2,0)
        image = image / 10000.0
        bboxes = []
        labels = []
        areas = []
        for annotation in annotations:
            bbox = annotation['bbox']
            if annotation['category_id'] != 0 and config.no_classes == 1:
                category_id = 1
            else:
                category_id = annotation['category_id']
            bboxes.append((bbox[0],bbox[1],bbox[2]+bbox[0],bbox[3]+bbox[1]))
            labels.append(category_id)
        print(labels[0])
        index = 0
        if labels[0] == 0:
            transformed = transform(image=image)
            transformed_2 = transform_3(image=image)
            image = transformed['image']
            _,h,w = image.shape
            target = {}
            target['boxes'] = torch.as_tensor(bboxes,dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels,dtype=torch.long)
            target['image_id'] = torch.tensor([index])
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["iscrowd"] = torch.ones((len(labels),), dtype=torch.int64)
        else:
            transformed = transform_bb(image=image, bboxes=bboxes, labels=labels)
            transformed_2 = transform_2(image=image, bboxes=bboxes, labels=labels)
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


    model = get_od_model(config).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device)["state_dict"])
    model.eval()
    print("target",target)
    targets ={"class_labels": target["labels"].to(device), "boxes": _voc_to_coco_boxes(target["boxes"].to(device))}
    print("image",image.shape)
    image = image.unsqueeze(0)
    with torch.no_grad():
        out = model(pixel_values=image)
        print(out)
    probs = out.logits.softmax(-1)[..., :-1]
    scores, pred_labels = probs.max(-1)
    preds_formatted = []
    for i in range(len(out.pred_boxes)):
        preds_formatted.append({
            "boxes": denormalize_xyxy(out.pred_boxes[i].detach().cpu()),
            "scores": scores[i].detach().cpu(),
            "labels": pred_labels[i].detach().cpu()
        })
    targets_formatted = {"boxes": _sanitize_boxes(target["boxes"].detach().cpu()), "labels": target["labels"].detach().cpu()}
    print("preds",preds_formatted)
    print("targets_formatted",targets_formatted)
    print("target",target)
    generate_composite_panel(transformed_2['image'], preds_formatted, targets_formatted, OUTDIR)

