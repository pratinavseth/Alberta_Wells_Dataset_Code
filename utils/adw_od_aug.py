import albumentations as A
from albumentations.pytorch import ToTensorV2

def object_detection_train_augmentation(config):
    """
    Create a composition of augmentations for the training phase.

    Args:
        config: Configuration object with preprocessing details.

    Returns:
        An Albumentations Compose instance with transformations.
    """
    return A.Compose([
        A.RandomSizedBBoxSafeCrop(height=256, width=256, erosion_rate=0.2),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.Normalize(mean=config.mean, std=config.std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format=config.bbox_format, min_visibility=0.1, label_fields=['labels'],clip=True)),A.Compose([
        A.RandomResizedCrop(height=256, width=256, scale=(0.9, 1.0)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.Normalize(mean=config.mean, std=config.std),
        ToTensorV2()
    ])

def object_detection_eval_augmentation(config):
    """
    Create a composition of augmentations for the evaluation phase.

    Args:
        config: Configuration object with preprocessing details.

    Returns:
        An Albumentations Compose instance with transformations.
    """
    return A.Compose([
        A.RandomSizedBBoxSafeCrop(height=256, width=256),
        A.Normalize(mean=config.mean, std=config.std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format=config.bbox_format, min_visibility=0.1, label_fields=['labels'],clip=True)),A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=config.mean, std=config.std),
        ToTensorV2()
    ])