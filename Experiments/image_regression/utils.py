import os
import math
import random

import cv2
import numpy as np
import einops
import kornia

import torch
from torchvision.utils import make_grid


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def grad2rgb(grad, q=0.05):
    """grad2rgb.
    The scale of grad do not affect the final rgb image

    :param grad: (b, h, w, 2)
    :param q: quantile
    """
    if grad.shape[-1] != 2:
        raise ValueError("grad is not of the right shape.") 
    B, H, W, _ = grad.shape
    grad_cpu = grad.detach().cpu()
    grad_x = grad_cpu[..., 0] # mGc, (b, h, w)
    grad_y = grad_cpu[..., 1] # mGr
    # grad_angle = torch.arctan(grad_x / grad_y) # (b, h, w), numerical instability
    grad_angle = torch.from_numpy(np.arctan2(grad_x.numpy(), grad_y.numpy()))
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)) # (b, h, w)
    grad_hsv = torch.zeros((B, 3, H, W), dtype=torch.float32) # (b, 3, h, w)
    grad_hsv[:, 0] = (grad_angle + torch.pi) # kornia.color.hsv_to_rgb assume Hue values are in the range [0 ~ 2pi]
    grad_hsv[:, 1] = 1.

    per_min = torch.quantile(einops.rearrange(grad_mag, 'b h w -> b (h w)'), q=q, dim=-1)
    per_max = torch.quantile(einops.rearrange(grad_mag, 'b h w -> b (h w)'), q=1.-q, dim=-1)

    grad_mag = (grad_mag - per_min) / (per_max - per_min)
    grad_mag = torch.clip(grad_mag, 0., 1.)

    grad_hsv[:, 2] = grad_mag
    grad_rgb = kornia.color.hsv_to_rgb(grad_hsv) # (b, 3, h, w)
    
    return grad_rgb


def cal_psnr(gt, pred, max_val=1.):
    mse = (gt - pred).pow(2).mean()
    return 10. * torch.log10(max_val ** 2 / mse)


def cal_ssim(gt, pred):
    return kornia.metrics.ssim(gt, pred, 11).mean()


def cal_metric(gt, pred, factor=None):
    gt = (gt * 0.5) + 0.5 # [-1. ~ 1.] to [0 ~ 1.]
    pred = (pred * 0.5) + 0.5

    if gt.shape[1] == 3: # color
        gt = (gt * torch.tensor([65.738, 129.057, 25.064]).div(256.).view(1, 3, 1, 1).to(gt.device)).sum(dim=1, keepdim=True) # (b, 1, h, w)
        pred = (pred * torch.tensor([65.738, 129.057, 25.064]).div(256.).view(1, 3, 1, 1).to(pred.device)).sum(dim=1, keepdim=True)

    if factor == None: # train
        psnr = cal_psnr(gt, pred)
        ssim = cal_ssim(gt, pred)
    else: # test
        eval_mask = torch.ones_like(gt, dtype=torch.bool) # (b, 1, H, W)
        eval_mask[:, :, ::factor, ::factor] = False
        psnr = cal_psnr(gt[eval_mask], pred[eval_mask]) # only evaluate test data
        ssim = cal_ssim(gt[..., factor:-factor, factor:-factor], pred[..., factor:-factor, factor:-factor]) # crop image following common settings of super resolution.

    return psnr, ssim


def write_train_summary(data, output, writer, epoch, shape, out_train_imgs_dir):
    prefix = 'train_'

    gt = data['downsampled_img'] # (h*w, c)
    gt = einops.rearrange(gt, '(h w) c -> c h w', h=shape.height).unsqueeze(0) # (1, c, h, w)
    pred = output['pred']
    pred = einops.rearrange(pred, '(h w) c -> c h w', h=shape.height).unsqueeze(0)

    # write image
    gt_vs_pred = torch.cat([gt, pred], dim=-1) # (b, c, h, w*2)
    gt_vs_pred = make_grid(gt_vs_pred, scale_each=True, normalize=True) # (c, h, w*2)
    writer.add_image(prefix + 'gt_vs_pred', gt_vs_pred, global_step=epoch)
    cv2.imwrite(os.path.join(out_train_imgs_dir, 'gt_vs_pred' + '-' + '%05d' % epoch + '.png'), (255 * einops.rearrange(torch.flip(gt_vs_pred, dims=[0]), 'c h w -> h w c').cpu().numpy()).astype(np.uint8)) 

    # write grad
    gt_grad = data['downsampled_grad'] # (h*w, c*2]
    gt_grad = einops.rearrange(gt_grad, '(h w) t -> h w t', h=shape.height).unsqueeze(0) # (1, h, w, c*2)
    pred_grad = output['pred_grad']
    pred_grad = einops.rearrange(pred_grad, '(h w) t -> h w t', h=shape.height).unsqueeze(0)

    if output['pred'].shape[1] == 1: # gray
        gt_grad_rgb = grad2rgb(gt_grad) # (b, c, h, w)
        pred_grad_rgb = grad2rgb(pred_grad)
        gt_vs_pred_grad = torch.cat([gt_grad_rgb, pred_grad_rgb], dim=-1) # (b, c, h, w*2)
        gt_vs_pred_grad = make_grid(gt_vs_pred_grad, scale_each=True, normalize=True) # (c, h, w*2)
        writer.add_image(prefix + 'gt_vs_pred_grad', gt_vs_pred_grad, global_step=epoch)
        cv2.imwrite(os.path.join(out_train_imgs_dir, 'gt_vs_pred_grad' + '-' + '%05d' % epoch + '.png'), (255 * einops.rearrange(torch.flip(gt_vs_pred_grad, dims=[0]), 'c h w -> h w c').cpu().numpy()).astype(np.uint8))

    elif output['pred'].shape[1] == 3: # color
        for i, c in enumerate(['r', 'g', 'b']):
            gt_grad_rgb = grad2rgb(gt_grad[..., 2*i:2*i+2])
            pred_grad_rgb = grad2rgb(pred_grad[..., 2*i:2*i+2])
            gt_vs_pred_grad = torch.cat([gt_grad_rgb, pred_grad_rgb], dim=-1)
            gt_vs_pred_grad = make_grid(gt_vs_pred_grad, scale_each=True, normalize=True)
            writer.add_image(prefix + c + '_gt_vs_pred_grad', gt_vs_pred_grad, global_step=epoch)
            cv2.imwrite(os.path.join(out_train_imgs_dir, c + '_gt_vs_pred_grad' + '-' + '%05d' % epoch + '.png'), (255 * einops.rearrange(torch.flip(gt_vs_pred_grad, dims=[0]), 'c h w -> h w c').cpu().numpy()).astype(np.uint8))

    # write metrics
    psnr, ssim = cal_metric(gt, pred)
    writer.add_scalar(prefix + 'PSNR', psnr, epoch)
    writer.add_scalar(prefix + 'SSIM', ssim, epoch)

    return psnr, ssim


def write_test_summary(data, output, writer, epoch, shape, out_test_imgs_dir, factor):
    prefix = 'test_'

    gt = data['img'].clone() # [h*w, c)
    gt = einops.rearrange(gt, '(h w) c -> c h w', h=shape.height).unsqueeze(0) # (1, c, h, w)
    pred = output['pred'].clone()
    pred = einops.rearrange(pred, '(h w) c -> c h w', h=shape.height).unsqueeze(0)

    # write img
    gt_vs_pred = torch.cat([gt, pred], dim=-1) # (b, c, h, w*2)
    gt_vs_pred = make_grid(gt_vs_pred, scale_each=True, normalize=True) # (c, h, 2*w)
    cv2.imwrite(os.path.join(out_test_imgs_dir, 'gt_vs_pred' + '-' + '%05d' % epoch + '.png'), (255 * einops.rearrange(torch.flip(gt_vs_pred, dims=[0]), 'c h w -> h w c').cpu().numpy()).astype(np.uint8))
    writer.add_image(prefix + 'gt_vs_pred', gt_vs_pred, global_step=epoch)

    # write metric
    psnr, ssim = cal_metric(gt, pred, factor)
    writer.add_scalar(prefix + 'PSNR', psnr, epoch)
    writer.add_scalar(prefix + 'SSIM', ssim, epoch)

    return psnr, ssim
