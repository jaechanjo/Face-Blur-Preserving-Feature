# -*- coding: utf-8 -*-
import argparse
import glob
import numpy as np
import os
import shutil
import random
import time
import datetime
import sys
import copy

import cv2
from tqdm import tqdm

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# %matplotlib inline

from utils.torch_utils import select_device
from models.experimental import attempt_load
from yolov5_face_detect import *

from cycleGAN import *
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as trn
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid

from torchvision import datasets
from albumentations.pytorch import ToTensorV2
import albumentations as A


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


def apply_blackbox(img, pos):
    """
    Apply black box
    :param img: PIL.Imagepip
    :param kernel: ker
    :param pos: list, normalized position for applying black box, [x1,y1,x2,y2]
    :return: PIL.Image
    """

    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    x1, y1, x2, y2 = pos
    x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
    face = image[y1:y2, x1:x2]
    face = np.zeros((face.shape[0], face.shape[1], 3), np.uint8)
    image[y1:y2, x1:x2] = face
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def face_blur_GAN_single(img, save_path=None, G_model_path='./weights/G_100_no.pth', Y_model_path='./weights/face_l.pt', transform=None,
                         img_size=640, conf_thres=0.5, iou_thres=0.5, device='0'):                           
    
    # image: dtype is np.array            

    # Load detector model
    device = select_device(device)
    model = attempt_load(Y_model_path, map_location=device)
    
    #yolov5_face_detect
    boxes = detect(model, img, img_size, conf_thres, iou_thres, device)
    
    # np.array-> PIL.Image                    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    blur = copy.deepcopy(img)
    o_h, o_w = img.size[1], img.size[0]
    
    #1개 얼굴만 크롭
    x1 = int(boxes[0][0])/o_w
    y1 = int(boxes[0][1])/o_h
    x2 = (int(boxes[0][0]) + int(boxes[0][2]))/o_w
    y2 = (int(boxes[0][1]) + int(boxes[0][3]))/o_h

    pos = [x1,y1,x2,y2]
    
    blur = apply_blackbox(blur, pos)
    
    # transform
    _transform = [
            A.Resize(224, 224),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2()
        ] if transform is None else transform
    
    transform = A.Compose(
        transforms=_transform,
        bbox_params=A.BboxParams(format='albumentations'),
        additional_targets={'blur': 'image'})
    
    
    t = transform(image=np.array(img), blur=np.array(blur), bboxes=[[*pos, 0]])
    
    blur = t['blur'].unsqueeze(0)
    
    image = t['image'].unsqueeze(0)
    
    # bboxes = [[[[t['bboxes'][0][0]], [t['bboxes'][0][1]], [t['bboxes'][0][2]], [t['bboxes'][0][3]]]]]
    bboxes = t['bboxes']

    if not len(t['bboxes']):
        bboxes = [(.0, .0, .0, .0, 0)]

    bboxes = torch.Tensor(bboxes[0])
    bboxes = bboxes.unsqueeze(0)
    
    # Load Generator model
    G = GeneratorResNet_no_residual_block(3)
    G = Wrapper(G)
    G = nn.DataParallel(G.cuda())
    G.load_state_dict(torch.load(G_model_path))
    G.eval()
    
    generated_img = G(blur, image, bboxes)
    generated_img = make_grid(generated_img, nrow=5, normalize=True)
    generated_img = transform_tensor_to_image(generated_img)
    generated_img = np.array(generated_img)
    generated_img = cv2.resize(generated_img, (o_w, o_h))

    if save_path is None:
        pass
    else:
        cv2.imwrite(save_path, cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR))

    return generated_img #np.array

# +
#test
# blur_img = face_blur_GAN_single(test_img, save_path='./data/result/blur_face1.jpg')
