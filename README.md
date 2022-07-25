# face-blur

# Jaechan Jo, Jongseok Lee, Sogang multi-media-system lab, Master's students

# feature inversion
'''multi images to multi faces of multi blur images '''

  1. command line interface
  ```python face_blur_feature_inversion.py --distort_weight 5, --fade_weight 2, --dataset_folder = './data/img/', --save_folder = None, --eval=False```

  2. function wrapping
  ```def face_blur_multi(distort_weight, fade_weight, dataset_folder, save_folder=None, weights='./weights/face_l.pt', img_size=640, conf_thres=0.5, iou_thres=0.5, iteration=400, device='0', eval = False):```

'''one image to multi faces of one blur image '''

  - function wrapping
  ```def face_blur_single(image, distort_weight, fade_weight, save_folder=None, weights='./weights/face_l.pt', img_size=640, conf_thres=0.5, iou_thres=0.5, iteration=400, device='0', eval = False):```

# cycleGAN
'''one image to only one face of one blur image '''

  - function wrapping
  ```def face_blur_GAN_single(img, save_path=None, G_model_path='./weights/G_100_no.pth', Y_model_path='./weights/face_l.pt', transform=None, img_size=640, conf_thres=0.5, iou_thres=0.5, device='0'):```
                         
                         
