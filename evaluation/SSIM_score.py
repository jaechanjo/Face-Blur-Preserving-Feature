# De-identification measurement SSIM #

from IQA_pytorch import SSIM, utils


def SSIM_score(generate_crop, original_crop, device):
    """
    Compute the face de-identification degree by SSIM score.
    
    Inputs:
    - generate_crop : 생성 blur된 얼굴 부분 이미지: PIL.Image size (W, H)
    - original_crop : 원본 얼굴 부분 이미지: PIL.Image size (W, H)
    
    Returns:
    - SSIM_score : 비식별화 지표: 0-1, 1에 가까울 수록 비슷한 품질을 나타낸다.
    """
    ori = utils.prepare_image(original_crop.convert("RGB")).to(device)
    gen = utils.prepare_image(generate_crop.convert("RGB")).to(device)
    
    #SSIM
    model = SSIM(channels=3)
    score = model(gen, ori, as_loss=False)
    
    return score.item()
