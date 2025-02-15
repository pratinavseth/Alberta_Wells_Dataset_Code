import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR,PolynomialLR

def get_lr_scheduler(config, optimizer, **kwargs):
    if config.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', config.patience),
            threshold=kwargs.get('threshold', 0.01),
            threshold_mode=kwargs.get('threshold_mode', 'rel'),
            cooldown=kwargs.get('cooldown', 0),
            min_lr=kwargs.get('min_lr', config.min_lr),
            eps=kwargs.get('eps', 1e-08)
        )
    elif config.lr_scheduler == "StepLR":
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif config.lr_scheduler == "ExponentialLR":
        scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=kwargs.get('gamma', 0.9)
        )
    elif config.lr_scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=kwargs.get('T_max', config.max_epoch),
            eta_min=kwargs.get('eta_min', config.min_lr)
        )
    elif config.lr_scheduler == "PolynomialLR":
        scheduler = PolynomialLR(
            optimizer=optimizer, 
            total_iters=kwargs.get('total_iters', config.max_epoch), 
            power=kwargs.get('power', config.lrs_power))
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler