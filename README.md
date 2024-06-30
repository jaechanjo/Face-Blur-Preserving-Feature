# Face-Blur

Official Pytorch implementation of Face-Blur Model | [Paper](https://drive.google.com/file/d/1WWVpRhUIyWVQBovvyZGyeGUG4fgMQLwm/view?usp=share_link), [PPT](https://drive.google.com/file/d/1PYEvL8hSJ6eYEbfVlRE6eTtc06w-99dg/view?usp=drive_link)

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
>   - ```image```**(single)**: image to make blur : content_image. (입력 이미지)
>   - ```distort_weight```**(1~5)**: As the weight increases, the face becomes increasingly distorted. (왜곡 가중치, 얼굴 왜곡 정도)
>   - ```fade_weight```**(1~5)**: As the weight increases, the face gradually fades. (희석 가중치, 얼굴 희미한 정도)
>   - ```dataset_folder```**(default)**: original content face images directory. (입력 이미지 폴더 경로)


> - *Additional_params*
>   - ```save_folder```: directory to save blur results. (결과 이미지 저장 폴더 경로)
>   - ```weights```: (face_detector)_model.pt path(s). (얼굴 탐지 모델 가중치 파일 경로)
>   - ```img_size```: (face_detector)_inference size (pixels). (얼굴 탐지 모델 입력 이미지 크기)
>   - ```conf_thres```: (face_detector)_object confidence threshold. (얼굴 탐지 확신 정도 기준 값)
>   - ```iou_thres```:  (face_detector)_IOU threshold for NMS. (얼굴 탐지 겹친 영역 정도 기준 값)
>   - ```iteration```: how many iterations to feature-inversion. (학습 횟수)
>   - ```device```: cuda device, i.e. 0 or 0,1,2,3 or cpu. (GPU 설정)
>   - ```eval```: show various evaluation tools : blur_image, inference_time, cos_similarity, SSIM. (평가 지표 출력 여부)
>   - ```circle```: blur face in shape of circle or rectangle, if False, its rectangle. (비식별화 모양, 타원형 | 직사각형)

### 2. CycleGAN

```Yolov5(face-detector) > CycleGAN(residual block | no residual block)```

### Hyper-parameter description

> - *Adjusted_params*
>   - ```img```: image to make blur : content_image


> - *Additional_params*
>   - ```save_path```: directory to save blur images. (결과 이미지 저장 폴더 경로)
>   - ```G_model_path```: Generator_model.pt path(s). (생성 모델 가중치 파일 경로)
>   - ```Y_model_path```: Yolov5(face_detector)_model.pt path(s). (얼굴 탐지 모델 가중치 파일 경로)
>   - ```transform```: To do transform. (이미지 사이즈, 정규화, 텐서 변환 여부)
>   - ```img_size```: (face_detector) inference size (pixels). (얼굴 탐지 모델 입력 이미지 크기)
>   - ```conf_thres```: (face_detector) object confidence threshold. (얼굴 탐지 확신 정도 기준 값)
>   - ```iou_thres```: (face_detector) IOU threshold for NMS. (얼굴 탐지 겹친 영역 정도 기준 값)
>   - ```device```: cuda device, i.e. 0 or 0,1,2,3 or cpu. (GPU 설정)
>   - ```residual_block```: whether the generator has residual block or not. (잔차 층 추가 여부)
>   - ```circle```: blur face in shape of circle or rectangle, if False, its rectangle. (비식별화 모양, 타원형 | 직사각형)


## Setup

### 1. Docker compose

```
$ cd ${WORKSPACE}  # git clone할 directory
$ git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/jaechanjo/Face-Blur-Preserving-Feature.git
$ cd Face-Blur-Preserving-Feature
$ docker-compose up -d  # docker 만들기
```

 > **git clone한 directory**에서 위 명령어를 실행합니다.

### 2. Packages

```
$ cat requirements.txt | while read PACKAGE; do pip install "$PACKAGE"; done
```

 > **install error**를 무시하고 설치합니다.

## Inference code


### Multi images with Multi-faces
> #### 다중 얼굴 이미지 폴더 단위

1. face_blur_feature_inversion.py
  
  ```
  $ python face_blur_feature_inversion.py \
  --distort_weight [int: 1~5] \  # 정수 1~5
  --fade_weight [int: 1~5] \  # 정수 1~5
  --dataset_folder [str: image path] \  # 이미지 폴더 경로
  --save_folder [None | str: save path] \  # 저장 폴더 경로
  --eval [boolean: True | False] \  # 평가지표 표시 여부
  --circle [boolean: True | False]  # 타원형 표시 여부
  ```

2. face_blur_feature_inversion_func.py

  ```
  $ def face_blur_multi(distort_weight=[int: 1-5], fade_weight=[int: 1-5], dataset_folder=[str: image path],  # 위와 동일
                      save_folder=[None | str: save path], weights=[str: yolov5 weight path], 
                      eval = [boolean: True | False], circle = [boolean: True | False])
  ```


### One images with Multi-faces
> #### 다중 얼굴 이미지 한 장 단위

3. face_blur_feature_inversion_func_single.py

  ```
  $ def face_blur_single(image, distort_weight=[int: 1-5], fade_weight=[int: 1-5], save_folder=[None | str: save path],  # 비식별화할 이미지(nd.array)
                       weights=[str: yolov5 weight path], eval = [boolean: True | False], circle = [boolean: True | False])
  ```


### One images with One-faces
> #### 단일 얼굴 이미지 한 장 단위

4. face_blur_GAN_func_single.py

  ```
  $ def face_blur_GAN_single(img, save_path=[None | str: save path],  # 비식별화할 이미지(nd.array)
                           G_model_path=[str: generator weight path], Y_model_path=[str: yolov5(face_detector) weight path],  # 생성모델, 얼굴 디텍터 가중치 경로
                           residual_block = [boolean: True | False], circle = [boolean: True | False])  # residual block 추가 여부 
  ```

  
## Reference


##### - Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "Image style transfer using convolutional neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
