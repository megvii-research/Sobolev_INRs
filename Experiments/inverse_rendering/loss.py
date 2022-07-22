import torch
import numpy as np

import diff_operators


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def der_mse(rgb, coordinate, gt_grad):

    pred_grad_r = diff_operators.gradient(
        rgb[..., 0], coordinate
    )  # [B, N, 2], order: (r_x, r_y)
    pred_grad_g = diff_operators.gradient(
        rgb[..., 1], coordinate
    )  # [B, N, 2], order: (g_x, g_y)
    pred_grad_b = diff_operators.gradient(
        rgb[..., 2], coordinate
    )  # [B, N, 2], order: (b_x, b_y)
    pred_grad = torch.concat(
        (pred_grad_r, pred_grad_g, pred_grad_b), dim=-1
    )  # [B, N, 6]

    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda()
    der_loss = torch.mean((weights * (gt_grad - pred_grad).pow(2)).sum(-1))

    return der_loss
