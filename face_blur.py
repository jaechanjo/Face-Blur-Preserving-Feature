# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
import PIL
import cv2
import argparse
import numpy as np
import os
import time
import copy
import glob
from tqdm import tqdm
from pathlib import Path
import torchvision
import torchvision.transforms as T
from yolov5_face_detect import *
from models.experimental import attempt_load
from utils.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from utils.extract_features import extract_features
from utils.torch_utils import select_device
from evaluation.feature_inversion_loss import face_loss
from evaluation.SSIM_score import SSIM_score
from imageio import imread
import matplotlib.pyplot as plt
import matplotlib.image as img
# %matplotlib inline

# +
# Check gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# multi-gpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# print('Device:', device, '/Name:', torch.cuda.get_device_name())
# print('Count of using GPUs:', torch.cuda.device_count())
# print('Current cuda device:', torch.cuda.current_device())

# +
# Load the pre-trained SqueezeNet model. (imagenet pretrained)
cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.to(device)

# We don't want to train the model any further, so we don't want PyTorch to waste computation 
# computing gradients on parameters we're never going to update.

for param in cnn.parameters():
    param.requires_grad = False


# +
def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),                                           
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def preprocess_rand(img):
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess_rand(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]), 
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x): # MINMAX Standardization
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


# -

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--distort_weight', nargs='?', type=int, default=1, help='(1~5) As the weight increases, the face becomes increasingly distorted.')
    parser.add_argument('--fade_weight', nargs='?', type=int, default=1, help='(1~5) As the weight increases, the face gradually fades.')
    parser.add_argument('--dataset_folder', type=str, help='original content face images directory')
    parser.add_argument('--save_folder', type=str, default=None, help='directory to save blur images')
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/face_l.pt', help='(face_detector)_model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='(face_detector)_inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='(face_detector)_object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='(face_detector)_IOU threshold for NMS')
    parser.add_argument('--iteration', type=int, default=400, help='how many iterations to feature-inversion')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval', type=str, default=False, help='show various evaluation tools : blur_image, inference_time, cos_similarity, de-identification value(SSIM)')
    parser.add_argument('--circle', type=bool, default=True, help='blur face in shape of circle or rectangle, if False, its rectangle')
    
    opt = parser.parse_args()
    
    if not opt.eval == 'False':
        #time check
        start_time = time.time()
    
    # Load model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    
    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 0.9*opt.iteration

    # Blur hyperparameters
    
    if opt.distort_weight > 5: # error exception
        opt.distort_weight = 5
    elif opt.distort_weight < 1:
        opt.distort_weight = 1
    
    if opt.fade_weight > 5: # error exception
        opt.fade_weight = 5
    elif opt.fade_weight < 1:
        opt.fade_weight = 1
        
    fade_dict = {1: 3e11, 2: 7e11, 3: 1e12, 4: 5e12, 5: 9e12}
    distort_dict = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9}
    target_layer = distort_dict[opt.distort_weight] #target_layer, the layer that do inverse feature from
    
    # testing dataset
    testset_folder = opt.dataset_folder
    
    #-------------------------------
    #synthesize random
    for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*'))):
        
        #image_name
        img_name = os.path.basename(image_path)[:-4]
        
        if not opt.eval == 'False':
            #time check
            infer_start_time = time.time()
        
        if image_path.endswith('.txt'):
            continue
        imgC = cv2.imread(image_path)  # imgC: Content-original_images
        imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB) #BGR -> RGB
        
        #Record original image size
        h, w, _ = imgC.shape
        
        if imgC is None:
            print(f'ignore : {image_path}')
            continue
        
        #yolov5_face_detect
        boxes = detect(model, imgC, opt.img_size, opt.conf_thres, opt.iou_thres, device) #conf > 0.5

        bbox_list = []
        
        imgB = copy.deepcopy(imgC) #Copy, imgB: Blur-images
        for box in boxes:

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[0]) + int(box[2])
            y2 = int(box[1]) + int(box[3])

            # bounding box list
            bbox_list.append( (x1,y1,x2,y2) )

            if opt.circle:
                ####### eclipse ########
                #center (x,y)
                x = (x1+x2)//2
                y = (y1+y2)//2

                #long_axis, short_axis
                l = round((y2-y1)*(9/16))
                s = round((x2-x1)*(5/8))

                #make eclipse mask
                mask = np.zeros((h,w), dtype=np.int8)
                cv2.ellipse(mask, (x, y), (l, s), 90, 0, 360, (1,1,1), -1)

                # delete face bounding circle
                imgB = imgB - imgB*mask[...,np.newaxis]

                # synthesize blur face circle
                random = np.array(np.random.uniform(low=0, high=255, size=(h, w, 3)), dtype='uint8')# 0~255
                imgB = imgB + random*mask[...,np.newaxis]
            
            else:
                ######rectangle######
                # delete face bbox
                imgB[y1:y2, x1:x2, :] -= imgB[y1:y2, x1:x2, :]

                #synthesize uniform random noise
                imgB[y1:y2, x1:x2, :] += np.array(np.random.uniform(low=0, high=255, size=(y2-y1, x2-x1, 3)), dtype='uint8')
                
        # numpy(cv2) -> torch.Tensor | Resize(512) | Normalize(squeeze_mean, squeeze_std)
        imgC = preprocess(PIL.Image.fromarray(imgC))
        imgB = preprocess(PIL.Image.fromarray(imgB.astype(np.uint8)))
        
        # Extract Zero(original)_image feature
        featsC = extract_features(imgC, cnn, device=device)
        
        #Assign imgB on auto_graph (parameter)
        imgB = imgB.to(device).requires_grad_()
        
        #Optimizer
        optimizer = torch.optim.Adam([imgB], lr=initial_lr)
        
        #Feature-Inversion in times of iteration
        for t in range(opt.iteration):
            
            #Initialize weight
            optimizer.zero_grad()
            
            #(lr_scheduling) 0.9*iteration이면, lr를 0.1으로 스케줄링
            if t == decay_lr_at:
                optimizer = torch.optim.Adam([imgB], lr=decayed_lr) 
            
            #(Clamping) 0.95*iteration이면, imgB weight bound in (-1.5, 1.5) -> nonlinearity like relu
            if t < 0.95*opt.iteration:
                imgB.data.clamp_(-1.5, 1.5)

            # Extract updated Blur_image feature
            featsB = extract_features(imgB, cnn, device=device)
            
            # Loss
            loss = face_loss(fade_dict[opt.fade_weight], featsB[target_layer], featsC[target_layer])
            
            # update the amount of loss
            loss.backward()
            optimizer.step()
        
        if not opt.eval == 'False':
            #Cos-sim btw blur_image, Content_image of target_layer
            cos = torch.nn.CosineSimilarity(dim=0)
            cos_sim = cos(torch.flatten(featsB[target_layer]), torch.flatten(featsC[target_layer]))

        #Blur_result (torch.Tensor -> np.array)& Back to original size
        imgB_result = T.functional.resize(deprocess(imgB.detach().cpu()), size=(h,w), interpolation= torchvision.transforms.functional.InterpolationMode.LANCZOS) # Auto_graph에서 내리자
        imgB_result = np.asarray(imgB_result) #asarray를 써서 데이터형태가 같은 경우, 메모리 같은 곳 사용(메모리 절약)
        
        #Blur faces+ Content background (torch.Tensor -> np.array)& Back to original size
        imgD = T.functional.resize(deprocess(imgC), size=(h,w), interpolation= torchvision.transforms.functional.InterpolationMode.LANCZOS) # Lanczos 보간법 (8x8 이웃 픽셀 참조)
        imgD = np.asarray(imgD)
        
        #De-identification btw Blur_faces, Content_faces
        de_identifi=0
        for bbox in bbox_list:
            
            if not opt.eval == 'False':
                #imgD == imgC for now
                de_identifi += 1 - SSIM_score(original_crop = PIL.Image.fromarray(imgD[bbox[1]:bbox[3], bbox[0]:bbox[2], :].astype(np.uint8)), 
                                              generate_crop = PIL.Image.fromarray(imgB_result[bbox[1]:bbox[3], bbox[0]:bbox[2], :].astype(np.uint8)),
                                              device=device)

            if opt.circle:
                ####### eclipse ######
                #center (x,y)
                x = (bbox[0]+bbox[2])//2
                y = (bbox[1]+bbox[3])//2

                #long_axis, short_axis
                l = round((y2-y1)*(9/16))
                s = round((x2-x1)*(5/8))

                #make eclipse mask
                mask = np.zeros((h,w), dtype=np.int8)
                cv2.ellipse(mask, (x, y), (l, s), 90, 0, 360, (1,1,1), -1)

                # delete face bounding circle
                imgD = imgD - imgD*mask[...,np.newaxis]

                # synthesize blur face circle
                imgD = imgD + imgB_result*mask[...,np.newaxis]
            
            else:
                ######rectangle######
                # delete face bbox
                imgD[bbox[1]:bbox[3], bbox[0]:bbox[2], :] -= imgD[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

                # synthesize blur result faces
                imgD[bbox[1]:bbox[3], bbox[0]:bbox[2], :] += imgB_result[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        
        # np.array -> PIL.Image
        imgD = PIL.Image.fromarray(imgD.astype(np.uint8))
        
        # Extract Blur faces+ Content images
        featsD = extract_features(preprocess(imgD), cnn, device=device)
        
        if not opt.eval == 'False':
            de_identification = de_identifi/len(bbox_list) #mean of faces

            #Cos-sim btw target_layer feature of squeezenet
            cos_sim_target = cos(torch.flatten(featsD[target_layer]), torch.flatten(featsC[target_layer]))
        
#         #Cos-sim btw output feature of squeezenet
#         cos_sim_out = cos(torch.flatten(featsD[-1]), torch.flatten(featsC[-1]))
    

            # Output(Content-original& Blur)
            f, axarr = plt.subplots(1,2, figsize=(10, 20))
            axarr[0].axis('off')
            axarr[1].axis('off')
            axarr[0].set_title('Original img.')
            axarr[1].set_title('Blur img.')
            axarr[0].imshow(deprocess(imgC)) # De-normalize (torch.Tensor -> PIL.Image)
            axarr[1].imshow(imgD) 
            plt.show()

            # 학습 정도-해당 층의 feature 유사도
            print(f"{img_name} 학습 정도-해당 층 feature (코사인)유사도(%): {cos_sim*100:.1f}")

            # 검색 성능 지표-CNN(squeezenet) output feature 유사도(imgD, imgC)
            print(f"{img_name} 검색 지표-CNN(squeezenet) feature (코사인)유사도(%): {cos_sim_target*100:.1f}")
            # 비식별화 값의 평균 (탐지된 얼굴들)
            print(f"{img_name} 비식별 값(1-SSIM)(%): {de_identification*100:.1f}")

            #Inference time of a image
            print(f"{img_name} 걸린 시간: {time.time() - infer_start_time: .2f}s")

            #Total time of images
            print(f'--------------------------------------------------')
            print(f'총 시간: {time.time() - start_time: .2f}s')
        else:
            cos_similarity = 0
            de_identification = 0
        
        #Save
        if opt.save_folder == 'None':
            pass
        else:
            os.makedirs(f"{opt.save_folder}{img_name}/", exist_ok=True)
            imgD.save(f'{opt.save_folder}{img_name}/blur_{img_name}-d{opt.distort_weight}-f{opt.fade_weight}-c{cos_sim_target*100:.1f}-s{de_identification*100:.1f}.jpg', 'JPEG')

