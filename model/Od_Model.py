import torchvision.models.detection as detection
import torchvision
import torch
    
def get_od_model(config):
    if config.architecture == "FasterRCNN":
        #no_classes =bg+no_of_classes
        if config.no_classes == 1:
            no_classes = config.no_classes+1
        else:
            no_classes = config.no_classes

        model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.backbone.body.conv1 = torch.nn.Conv2d(config.in_channels, 64, 
                                                    kernel_size=(7, 7), stride=(2, 2), 
                                                    padding=(3, 3), bias=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, no_classes)
        if config.in_channels == 3:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0],
                image_std=[1,1,1]
            )
        else: 
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0,0],
                image_std=[1,1,1,1]
            )
    
    elif config.architecture == "FasterRCNN_v2":
        #no_classes =bg+no_of_classes
        if config.no_classes == 1:
            no_classes = config.no_classes+1
        else:
            no_classes = config.no_classes

        model = detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
        model.backbone.body.conv1 = torch.nn.Conv2d(config.in_channels, 64, 
                                                    kernel_size=(7, 7), stride=(2, 2), 
                                                    padding=(3, 3), bias=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, no_classes)
        if config.in_channels == 3:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0],
                image_std=[1,1,1]
            )
        else:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0,0],
                image_std=[1,1,1,1]
            )

    elif config.architecture == "FasterRCNN_v3":
        #no_classes =bg+no_of_classes
        if config.no_classes == 1:
            no_classes = config.no_classes+1
        else:
            no_classes = config.no_classes

        model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        model.backbone.body['0'][0] = torch.nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, no_classes)
        if config.in_channels == 3:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0],
                image_std=[1,1,1]
            )
        else:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0,0],
                image_std=[1,1,1,1]
            )

    elif config.architecture == "RetinaNet":
        #no_classes =bg+no_of_classes
        if config.no_classes == 1:
            no_classes = config.no_classes+1
        else:
            no_classes = config.no_classes
        
        model = detection.retinanet_resnet50_fpn(num_classes = no_classes,pretrained_backbone =True)
        model.backbone.body.conv1 = torch.nn.Conv2d(config.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if config.in_channels == 3:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0],
                image_std=[1,1,1]
            )
        else:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0,0],
                image_std=[1,1,1,1]
            )
    
    elif config.architecture == "FCOS":
        #no_classes =bg+no_of_classes
        if config.no_classes == 1:
            no_classes = config.no_classes+1
        else:
            no_classes = config.no_classes
        
        model = detection.fcos_resnet50_fpn(num_classes = no_classes,pretrained_backbone =True)
        if config.in_channels == 3:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0],
                image_std=[1,1,1]
            )
        else:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0,0],
                image_std=[1,1,1,1]
            )
        print(model)
        print(model.backbone.body.conv1)
        model.backbone.body.conv1 = torch.nn.Conv2d(config.in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)


    elif config.architecture == "SSD_Lite":
        #no_classes =bg+no_of_classes
        if config.no_classes == 1:
            no_classes = config.no_classes+1
        else:
            no_classes = config.no_classes
        
        model = detection.ssdlite320_mobilenet_v3_large(num_classes = no_classes,pretrained_backbone =True)
        if config.in_channels == 3:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=256,
                max_size=256,
                image_mean=[0,0,0],
                image_std=[1,1,1]
            )
        else:
            model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=256,
            max_size=256,
            image_mean=[0,0,0,0],
            image_std=[1,1,1,1]
        )
        model.backbone.features[0][0][0] = torch.nn.Conv2d(config.in_channels, 16, kernel_size=(3,3), stride=(2, 2), padding=(1,1), bias=False)


    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
    
    return model