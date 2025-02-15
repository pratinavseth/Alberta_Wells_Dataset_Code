import segmentation_models_pytorch as smp
from .Upernet import UperNet
from .SegFormer import Segformer

def get_seg_model(config):
    if config.architecture == "UperNet":
        model = UperNet(config)
    elif config.architecture == "Segformer":
        model = Segformer(config)

    elif config.architecture == "UNet":
        if str(config.activation_last_layer) != "None": 
            activation = str(config.activation_last_layer)
        else:
            activation = None
        model = smp.Unet(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=activation,
)
    elif config.architecture == "UNet++":
        if str(config.activation_last_layer) != "None": 
            activation = str(config.activation_last_layer)
        else:
            activation = None
        model = smp.UnetPlusPlus(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=activation,
)
    elif config.architecture == "MAnet":
        model = smp.MAnet(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=str(config.activation_last_layer),
)
    elif config.architecture == "Linknet":
        model = smp.Linknet(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=str(config.activation_last_layer),
)
    elif config.architecture == "FPN":
        model = smp.FPN(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=str(config.activation_last_layer),
)
    elif config.architecture == "PSPNet":
        if str(config.activation_last_layer) != "None": 
            activation = str(config.activation_last_layer)
        else:
            activation = None
        model = smp.PSPNet(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=activation,
)
    elif config.architecture == "PAN":
        if str(config.activation_last_layer) != "None": 
            activation = str(config.activation_last_layer)
        else:
            activation = None
        model = smp.PAN(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=activation,
)
    elif config.architecture == "DeepLabV3":
        model = smp.DeepLabV3(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=str(config.activation_last_layer),
)
    elif config.architecture == "DeepLabV3+":
        if str(config.activation_last_layer) != "None": 
            activation = str(config.activation_last_layer)
        else:
            activation = None
        model = smp.DeepLabV3Plus(
     encoder_name=str(config.backbone),
     encoder_weights=str(config.wt_initialization),
     in_channels=config.in_channels,
     classes=config.no_classes,
     activation=activation,
)
    return model