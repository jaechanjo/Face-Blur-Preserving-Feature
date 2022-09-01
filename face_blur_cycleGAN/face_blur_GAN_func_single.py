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


# 직시각형으로 얼굴 크롭해서 검정 박스로
def apply_blackbox_rect(img, pos):
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


# 타원형으로 얼굴 크롭해서 검정 박스로
def apply_blackbox_circle(img, pos):
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
    
    #center (x,y)
    x = (x1+x2)//2
    y = (y1+y2)//2
    
    #long_axis, short_axis
    l = round((y2-y1)*(9/16))
    s = round((x2-x1)*(5/8))
    
    #make eclipse mask
    mask = np.zeros((h,w), dtype=np.int8)
    cv2.ellipse(mask, (x, y), (l, s), 90, 0, 360, (1,1,1), -1)
    
    #delete face bounding circle
    image = image - image*mask[...,np.newaxis]
    
    #nd.array -> PIL.Image
    image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def face_blur_GAN_single(img, save_path=None, G_model_path='./weights/G_100_no.pth', Y_model_path='./weights/face_l.pt', transform=None,
                         img_size=640, conf_thres=0.5, iou_thres=0.5, device='0', residual_block=False, circle=True):                           
    
    '''
    Explain

    - Adjusted_params
    > > img: type=numpy.ndarray help=image to make blur : content_image


    - Additional_params
    > > save_path: type=str, default=None, help=directory to save blur images
    > > G_model_path: type=str, default='./weights/G_100_no.pth', help=(Generator)_model.pt path(s)
    > > Y_model_path: type=str, default='./weights/face_l.pt', help=(Yolov5)_model.pt path(s)
    > > transform: type=str, default=None, help=To do transform
    > > img_size: type=int, default=640, help=(face_detector)_inference size (pixels)
    > > conf_thres: type=float, default=0.5, help=(face_detector)_object confidence threshold
    > > iou_thres:  type=float, default=0.5, help=(face_detector)_IOU threshold for NMS
    > > device: type=str, default='0', help=cuda device, i.e. 0 or 0,1,2,3 or cpu
    > > residual_block: type=bool, default=False, help= whether the generator has residual block or not
    > > circle: type=bool, default=True, help=blur face in shape of circle or rectangle, if False, its rectangle
    '''

    # Load detector model
    device = select_device(device)
    model = attempt_load(Y_model_path, map_location=device)
    
    #yolov5_face_detect
    boxes = detect(model, img, img_size, conf_thres, iou_thres, device)
    
    # np.array-> PIL.Image                    
    img = Image.fromarray(img)
    blur = copy.deepcopy(img)
    o_h, o_w = img.size[1], img.size[0]
    
    #1개 얼굴만 크롭
    x1 = int(boxes[0][0])/o_w
    y1 = int(boxes[0][1])/o_h
    x2 = (int(boxes[0][0]) + int(boxes[0][2]))/o_w
    y2 = (int(boxes[0][1]) + int(boxes[0][3]))/o_h

    pos = [x1,y1,x2,y2]
    
    if circle:
        #crop in shape of circle
        blur = apply_blackbox_circle(blur, pos)
    else:
        #crop in shape of rectangle
        blur = apply_blackbox_rect(blur, pos)
    
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
    
    if residual_block:
        #exist residual block, G_100.pth
        G = GeneratorResNet(3, 9)
        G = Wrapper(G)
        G = nn.DataParallel(G.cuda())
        G.load_state_dict(torch.load(G_model_path))
        G.eval()
    else:
        # Load Generator model - no residual block, G_100_no.pth
        G = GeneratorResNet_no_residual_block(3)
        G = Wrapper(G)
        G = nn.DataParallel(G.cuda())
        G.load_state_dict(torch.load(G_model_path))
        G.eval()
    
    generated_img = G(blur, image, bboxes, circle)
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
# test_img = cv2.imread('./data/img/face2.jpg')
# test_img=cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# plt.imshow(test_img)
# plt.show()

# +
#test
# blur_img = face_blur_GAN_single(test_img, save_path='./data/result/blur_face2_no_circle.jpg', G_model_path='./weights/G_100_no.pth', residual_block=False, circle=True)

# +
# plt.imshow(blur_img)
# plt.show()
