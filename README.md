# face-blur

### De-identify the face while preserving image feature using feature inversion, cycleGAN
#### Jaechan Jo, Jongseok Lee, Sogang multi-media-system lab

## model structure
1. feature inversion: Yolov5(face-detector) > SqueezeNet(CNN)_feature inversion
2. cycleGAN: Yolov5(face-detector) > GAN with residual block
                                   > GAN with no residual block

## feature inversion

### hyper-parameter description

> Adjusted_params
> > - image(single): type=numpy.ndarray help=image to make blur : content_image
> > - distort_weight: type=int, default=1, help=(1~5) As the weight increases, the face becomes increasingly distorted.
> > - fade_weight: type=int, default=1, help=(1~5) As the weight increases, the face gradually fades.
> > - dataset_folder(multi): type=str, help=original content face images directory


> Additional_params
> > - save_folder: type=str, help=directory to save blur results
> > - weights: type=str, default='./weights/face_l.pt', help=(face_detector)_model.pt path(s)
> > - img_size: type=int, default=640, help=(face_detector)_inference size (pixels)
> > - conf_thres: type=float, default=0.5, help=(face_detector)_object confidence threshold
> > - iou_thres:  type=float, default=0.5, help=(face_detector)_IOU threshold for NMS
> > - iteration, type=int, default=400, help=how many iterations to feature-inversion
> > - device, type=str, default='0', help=cuda device, i.e. 0 or 0,1,2,3 or cpu
> > - eval, type=str, default=False, help=show various evaluation tools : blur_image, inference_time, cos_similarity, de-identification value(SSIM)

1. multi-images to multi-blur images with multi-faces

  - command line interface(face_blur_feature_inversion.py)<br><br>
  
  ```python face_blur_feature_inversion.py --distort_weight 5, --fade_weight 2, --dataset_folder = './data/img/', --save_folder = None, --eval=False```

- function wrapping(face_blur_feature_inversion_func_ver.py)<br><br>
  ```def face_blur_multi(distort_weight, fade_weight, dataset_folder, save_folder=None, weights='./weights/face_l.pt', img_size=640, conf_thres=0.5, iou_thres=0.5, iteration=400, device='0', eval = False):```

2. one-image to one-blur image with multi-faces

  - function wrapping(face_blur_feature_inversion_func_single.py)<br><br>
  ```def face_blur_single(image, distort_weight, fade_weight, save_folder=None, weights='./weights/face_l.pt', img_size=640, conf_thres=0.5, iou_thres=0.5, iteration=400, device='0', eval = False):```

## cycleGAN
3. one-image to one-blur image with one-face

  - function wrapping(face_blur_GAN_func_single.py)<br><br>
  ```def face_blur_GAN_single(img, save_path=None, G_model_path='./weights/G_100_no.pth', Y_model_path='./weights/face_l.pt', transform=None, img_size=640, conf_thres=0.5, iou_thres=0.5, device='0'):```
                         
                         
