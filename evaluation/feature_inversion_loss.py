# feature inversion loss #

import torch


def face_loss(blur_weight, start_img, original_img):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - blur_weight: 비식별화 정도; 숫자가 커지면 커질수록 학습이 약해져,비식별화 정도가 강해진다
    - start_img : 비식별화하고자 하는 얼굴 부분에 Random noise를 준 이미지 feature ; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - original_img : 비식별화하고자 하는 얼굴 부분이 포함된 전체 이미지 feature ; Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - feature_loss: SSE(Squared Sum Error) with inverse of blur_weight
    """
    feature_loss = (1/blur_weight) * torch.sum((torch.pow(start_img - original_img, 2)))
    
    return feature_loss
