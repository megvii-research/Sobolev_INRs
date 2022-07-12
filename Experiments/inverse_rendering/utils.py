import random
import numpy as np
import kornia

import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_psnr(gt, pred, max_val=1.):
    """cal_psnr.

    :param pred: [B, C, H, W]
    :param gt: [B, C, H, W]
    """
    mse = (gt - pred).pow(2).mean(dim=(1, 2, 3)) # [B]
    return 10. * torch.log10(max_val ** 2 / mse)


def cal_ssim(gt, pred):
    """cal_ssim.

    :param pred: [B, C, H, W]
    :param gt: [B, C, H, W]
    """
    return kornia.metrics.ssim(gt, pred, 11).mean(dim=(1, 2, 3)) # [B]
