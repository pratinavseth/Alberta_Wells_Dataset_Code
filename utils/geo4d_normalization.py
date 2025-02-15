import torch
from kornia.augmentation import AugmentationBase2D

class Standarize(AugmentationBase2D):
    def __init__(self, max_pixel_value=(14549.0, 18955.0, 24230.0, 46966.0)):
        super(Standarize, self).__init__(p=1.0)
        self.max_pixel_value = torch.tensor(max_pixel_value)

    def apply_transform(self, input, flags=None, params=None, transform=None):
        max_pixel_value = 1.0 / self.max_pixel_value.to(input.device)
        input = input.clone()
        input = input * max_pixel_value[:, None, None]
        return input

class Standarize_and_Normalize(AugmentationBase2D):
    def __init__(self,
                 mean=(0.03006608391496679, 0.03802018905804906, 0.0296749500591402, 0.05505464739747856),
                 std=(0.019981467499910793, 0.017820867319533553, 0.0195589033009335, 0.016053728918243495),
                 max_pixel_value=(14549.0, 18955.0, 24230.0, 46966.0)):
        super(Standarize_and_Normalize, self).__init__(p=1.0)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.max_pixel_value = torch.tensor(max_pixel_value)

    def apply_transform(self, input, flags=None, params=None, transform=None):
        max_pixel_value = 1.0 / self.max_pixel_value.to(input.device)
        input = input.clone()
        input = input * max_pixel_value[:, None, None]
        input = (input - self.mean[:, None, None].to(input.device)) / self.std[:, None, None].to(input.device)
        return input

class zNormalize(AugmentationBase2D):
    def __init__(self,
                 mean=(0.042566365753578035, 0.0713053507520663, 0.07111980741387171, 0.2528623366198003),
                 std=(0.036764543867300215, 0.04541646657693727, 0.05960256952003561, 0.07440134434546278)):
        super(zNormalize, self).__init__(p=1.0)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def apply_transform(self, input, flags=None, params=None, transform=None):
        input = input.clone()
        input = (input - self.mean[:, None, None].to(input.device)) / self.std[:, None, None].to(input.device)
        return input
