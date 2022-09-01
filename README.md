# Face-Blur

Official Pytorch implementation of Face-Blur Model | [Paper](https://drive.google.com/file/d/14Krwy55_S4TZ3VeRlEemMd-R0gWAk0ml/view?usp=sharing)

**[Jaechan Jo](mailto:jjc123a@naver.com), Jongseok Lee.**

Multi Media System Lab, Sogang AI Research.

## Sample Results
### Overview
De-identify the face while preserving image feature using feature inversion, cycleGAN

- Feature Inversion

  - <img width="300" alt="teaser" src="./data/result/blur_eunbin.jpg">

- CycleGAN
  - residual block

    <img width="300" alt="teaser" src="./face_blur_cycleGAN/data/result/blur_face1_circle.jpg">


  - no residual block

    <img width="300" alt="teaser" src="./face_blur_cycleGAN/data/result/blur_face1_no_circle.jpg">
  

## Model

### 0. Weight

- [CycleGAN-residual block: G_100.pth](https://drive.google.com/file/d/1SDpJphwBbtVAnOwDXUOJBqBIMTDuGK9b/view?usp=sharing)
- [CycleGAN-no residual block: G_100_no.pth](https://drive.google.com/file/d/1nWka8iwygB4uRXHty2x9JUSeCsyRlf9R/view?usp=sharing)

> Source from [GitHub - deepcam-cn/yolov5-face: YOLO5Face: Why Reinventing a Face Detector (https://arxiv.org/abs/2105.12931)](https://github.com/deepcam-cn/yolov5-face)
> - [Yolov5 face-large: face_l.pt](https://drive.google.com/file/d/1uWR7O4ka6dJitWLc9zR3kwFJhpsmCeqj/view?usp=sharing)
> - [Yolov5 face-medium: face_m.pt](https://drive.google.com/file/d/1blTdj5GXR8T5RoWGnNDXdx5ljPEag3Bh/view?usp=sharing)


### 1. Feature Inversion

```Yolov5(face-detector) > SqueezeNet(CNN-feature inversion)```

### Hyper-parameter description

> - *Adjusted_params*
>   - ```image```**(single)**: image to make blur : content_image
>   - ```distort_weight```**(1~5)**: As the weight increases, the face becomes increasingly distorted.
>   - ```fade_weight```**(1~5)**: As the weight increases, the face gradually fades.
>   - ```dataset_folder```**(default)**: original content face images directory


> - *Additional_params*
>   - ```save_folder```: type=str, help=directory to save blur results
>   - ```weights```: type=str, default='./weights/face_l.pt', help=(face_detector)_model.pt path(s)
>   - ```img_size```: type=int, default=640, help=(face_detector)_inference size (pixels)
>   - ```conf_thres```: type=float, default=0.5, help=(face_detector)_object confidence threshold
>   - ```iou_thres```:  type=float, default=0.5, help=(face_detector)_IOU threshold for NMS
>   - ```iteration```: type=int, default=400, help=how many iterations to feature-inversion
>   - ```device```: type=str, default='0', help=cuda device, i.e. 0 or 0,1,2,3 or cpu
>   - ```eval```: type=str, default=False, help=show various evaluation tools : blur_image, inference_time, cos_similarity, de-identification value(SSIM)


### 2. CycleGAN

```Yolov5(face-detector) > CycleGAN(residual block | no residual block)```

### Hyper-parameter description

> - *Adjusted_params*
>   - ```img```: image to make blur : content_image


> - *Additional_params*
>   - ```save_path```: directory to save blur images
>   - ```G_model_path```: Generator_model.pt path(s)
>   - ```Y_model_path```: Yolov5(face_detector)_model.pt path(s)
>   - ```transform```: To do transform
>   - ```img_size```: (face_detector) inference size (pixels)
>   - ```conf_thres```: (face_detector) object confidence threshold
>   - ```iou_thres```: (face_detector) IOU threshold for NMS
>   - ```device```: cuda device, i.e. 0 or 0,1,2,3 or cpu


## Setup

### Docker compose

```docker run --gpus all -itd -e LC_ALL=C.UTF-8 --name face_blur -v"[gpu server dir]":/workspace/ -p 20000:8888 -p 20001:8097 -p 20002:22 nvcr.io/nvidia/pytorch:21.07-py3 /bin/bash```

  > - **docker name(이름 정의)**: e.g.) face_blur
  > - **gpu server dir(도커 가상환경에 연결할 GPU 서버 폴더 경로)**: git clone dir(깃클론한 폴더 경로를 넣어주세요) e.g.) /media/mmlab/hdd3/Face-Blur-Preserving-Feature 
  > - **mounted docker dir(연결된 도커 폴더 경로)**: e.g.) /workspace/
  > - **port forwading(포트 설정)**: e.g.) 20000:8888(jupyter), 20001:8097(visdom), 20002:22(ssh)
  > - **docker image(도커 이미지)**: e.g.) nvcr.io/nvidia/pytorch:21.07-py3


## Inference code


### Multi images with Multi-faces

1. face_blur_feature_inversion.py
  
  ```
  python face_blur_feature_inversion.py \
  --distort_weight [int: 1~5] \
  --fade_weight [int: 1~5] \
  --dataset_folder [str: image path] \
  --save_folder [None | str: save path] \
  --eval [boolean: True | False]
  ```

2. face_blur_feature_inversion_func.py

  ```
  def face_blur_multi(distort_weight=[int: 1-5], fade_weight=[int: 1-5], dataset_folder=[str: image path],
  save_folder=[None | str: save path], weights=[str: yolov5 weight path], eval = [boolean: True | False])
  ```


### One images with Multi-faces

3. face_blur_feature_inversion_func_single.py

  ```
  def face_blur_single(image, distort_weight=[int: 1-5], fade_weight=[int: 1-5], save_folder=[None | str: save path],
  weights=[str: yolov5 weight path], eval = [boolean: True | False])
  ```

4. face_blur_feature_inversion_func_single.ipynb

  > it contains the results of the execution


### One images with One-faces


5. face_blur_GAN_func_single.py

  ```
  def face_blur_GAN_single(img, save_path=[None | str: save path],
  G_model_path=[str: generator weight path], Y_model_path=[str: yolov5(face_detector) weight path])
  ```
  
