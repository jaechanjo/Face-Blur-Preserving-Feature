import torch.nn as nn
import torch.nn.functional as F
import torch

from pathlib import Path
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Stitching_rect(nn.Module):
    def __init__(self):
        super(Stitching_rect, self).__init__()

    def forward(self, x, ori, bbox):
        h, w = x.shape[-2:]
        bbox[:, 0] = bbox[:, 0] * h
        bbox[:, 1] = bbox[:, 1] * w
        bbox[:, 2] = bbox[:, 2] * h
        bbox[:, 3] = bbox[:, 3] * w

        out = ori.clone()
        for n, (_x, _o, box) in enumerate(zip(x, ori, bbox)):
            x1, y1, x2, y2, _ = list(map(int, box))
            out[n, ..., y1:y2, x1:x2] = _x[..., y1:y2, x1:x2].clone()
        return out


class Stitching_circle(nn.Module):
    def __init__(self):
        super(Stitching_circle, self).__init__()

    def forward(self, x, ori, bbox):
        h, w = x.shape[-2:]
        bbox[:, 0] = bbox[:, 0] * h
        bbox[:, 1] = bbox[:, 1] * w
        bbox[:, 2] = bbox[:, 2] * h
        bbox[:, 3] = bbox[:, 3] * w
        
        out = ori.clone()
        
        transform = transforms.ToTensor()
        
        for n, (_x, _o, box) in enumerate(zip(x, ori, bbox)):
            x1, y1, x2, y2, _ = list(map(int, box))
            
            #center (x,y)
            x = (x1+x2)//2
            y = (y1+y2)//2
            
            #long_axis, short_axis
            l = round((y2-y1)*(9/16))
            s = round((x2-x1)*(5/8))
            
            #make eclipse mask
            mask = np.zeros((h,w), dtype=np.int8)
            cv2.ellipse(mask, (x, y), (l, s), 90, 0, 360, (1,1,1), -1)
            
            #make background mask
            mask_bg = np.ones((h,w), dtype=np.int8)
            cv2.ellipse(mask_bg, (x, y), (l, s), 90, 0, 360, (0,0,0), -1)
            
            #origin background mask + blur face mask
            out[n, ...] = _o.clone()*transform(mask_bg).cuda().unsqueeze(0)+ _x.clone()*transform(mask).cuda().unsqueeze(0)
        return out


class Wrapper(nn.Module):
    def __init__(self, generator):
        super(Wrapper, self).__init__()
        self.generator = generator
        self.stitching_rect = Stitching_rect()
        self.stitching_circle = Stitching_circle()

    def forward(self, img, origin, bbox, circle):
        g = self.generator(img)
        if circle:
            x = self.stitching_circle(g, origin, bbox)
        else:
            x = self.stitching_rect(g, origin, bbox)
        return x


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return summary(self, input_size, batch_size, device)
        except:
            return self.__repr__()


class GeneratorResNet_no_residual_block(nn.Module):
    def __init__(self, input_shape):
        super(GeneratorResNet_no_residual_block, self).__init__()

        channels = input_shape

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return summary(self, input_size, batch_size, device)
        except:
            return self.__repr__()


def transform_tensor_to_image(tensor):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    return im
