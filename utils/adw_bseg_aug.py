import torch
import kornia.augmentation as K
import kornia.geometry.transform as KG
from utils.geo4d_normalization import zNormalize, Standarize, Standarize_and_Normalize

class SegmentationTransform:
    def __init__(self, image_transforms, mask_transforms):
        self.image_transforms = torch.nn.ModuleList(image_transforms)
        self.mask_transforms = torch.nn.ModuleList(mask_transforms)

    def __call__(self, image, mask):
        for img_transform, mask_transform in zip(self.image_transforms, self.mask_transforms):
            if isinstance(img_transform, (K.RandomHorizontalFlip, K.RandomVerticalFlip, K.RandomRotation, K.RandomCrop)):
                params = img_transform.forward_parameters(image.shape)
                flags = img_transform.flags
                image = img_transform.apply_transform(input=image, params=params,flags=flags)
                mask = mask_transform.apply_transform(input=mask, params=params,flags=flags)
            else:
                image = img_transform(image)
                mask = mask_transform(mask)
        return image, mask

def bseg_train_augmentation(config):
    """
    Create a composition of augmentations for the final training phase.

    Args:
        config: Configuration object with preprocessing details.

    Returns:
        A SegmentationTransform instance with Kornia transformations.
    """
    
    image_transforms = [
        K.RandomResizedCrop(size=(256, 256),scale=(0.9,1.0)),
        K.RandomHorizontalFlip(p=0.25),
        K.RandomVerticalFlip(p=0.25)
    ]
    mask_transforms = [
        K.RandomResizedCrop(size=(256, 256),scale=(0.9,1.0)),
        K.RandomHorizontalFlip(p=0.25),
        K.RandomVerticalFlip(p=0.25)
    ]

    if config.pre_processing == "Standarize":
        image_transforms.append(Standarize(max_pixel_value=config.channel_max))
    elif config.pre_processing == "Standarize_and_Normalize":
        image_transforms.append(Standarize_and_Normalize(max_pixel_value=config.channel_max, mean=config.mean, std=config.std))
    else:
        image_transforms.append(K.Normalize(mean=config.mean, std=config.std))
        
    return SegmentationTransform(image_transforms, mask_transforms)

def bseg_eval_augmentation(config):
    """
    Create a composition of augmentations for the evaluation phase.

    Args:
        config: Configuration object with preprocessing details.

    Returns:
        A SegmentationTransform instance with Kornia transformations.
    """
    image_transforms = [
        KG.Resize((256, 256))
    ]
    mask_transforms = [
        KG.Resize((256, 256))
    ]

    if config.pre_processing == "Standarize":
        image_transforms.append(Standarize(max_pixel_value=config.channel_max))
    elif config.pre_processing == "Standarize_and_Normalize":
        image_transforms.append(Standarize_and_Normalize(max_pixel_value=config.channel_max, mean=config.mean, std=config.std))
    else:
        image_transforms.append(K.Normalize(mean=config.mean, std=config.std))
    
    return SegmentationTransform(image_transforms, mask_transforms)
