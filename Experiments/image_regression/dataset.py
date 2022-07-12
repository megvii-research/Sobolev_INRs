import os
from collections import namedtuple

from PIL import Image
import kornia
import einops

import torch
from torchvision import transforms


def get_data(data_root, dataset, filename, is_gray, factor, der_operator):
    
    dataset_root = os.path.join(data_root, dataset)

    img = Image.open(os.path.join(dataset_root, 'HR', filename + '.png'))
    if (img.height % factor) or (img.width % factor):
        raise ValueError("The width/height of image must be an integer multiple of factor!")
    if is_gray:
        img = img.convert('L')

    transform = transforms.Compose([
        transforms.Resize((img.height, img.width)),
        transforms.ToTensor(), # [0 ~ 255] to [0. ~ 1.]
        transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])) # [0. ~ 1.] to [-1. ~ 1.]
    ])
    img = transform(img) # (c, h, w)

    img_shape = namedtuple('shape', 'height width')(img.shape[1], img.shape[2])
    grad = kornia.filters.spatial_gradient(img.unsqueeze(0), mode=der_operator, order=1, normalized=True).squeeze(0) # (c, 2, h, w), grad_x = grad[:, 0], grad_y = grad[:, 1]
    coordinate = torch.stack(
            torch.meshgrid(
                torch.linspace(0, img.shape[2] - 1, img.shape[2]), # (0 ~ w-1)
                torch.linspace(0, img.shape[1] - 1, img.shape[1]), # (0 ~ h-1)
                indexing='xy'),
            -1) # (h, w, 2)

    downsampled_img = img[:, ::factor, ::factor].clone() # equal to F.interpolate with mode='nearest'
    # downsampled_img = F.interpolate(img.unsqueeze(0), scale_factor=1/factor, mode='nearest').squeeze(0)
    downsampled_img_shape = namedtuple('shape', 'height width')(downsampled_img.shape[1], downsampled_img.shape[2])
    downsampled_grad = grad[:, :, 0::factor, 0::factor].clone()
    downsampled_coordinate = coordinate[::factor, ::factor, :].clone()

    # reshape data
    img = einops.rearrange(img, 'c h w -> (h w) c') # (h*w, c)
    grad = einops.rearrange(grad, 'c d h w -> (h w) (c d)') # (h*w, c*2), if c=3: (dx_r, dy_r, dx_g, dy_g, dx_b, dy_b)
    coordinate = einops.rearrange(coordinate, 'h w c -> (h w) c') # (h*w, 2)

    downsampled_img = einops.rearrange(downsampled_img, 'c h w -> (h w) c')
    downsampled_grad = einops.rearrange(downsampled_grad, 'c d h w -> (h w) (c d)')
    downsampled_coordinate = einops.rearrange(downsampled_coordinate, 'h w c -> (h w) c')


    return {
            'img': img, 
            'img_shape': img_shape,
            'grad': grad,
            'coordinate': coordinate,

            'downsampled_img': downsampled_img,
            'downsampled_img_shape': downsampled_img_shape,
            'downsampled_grad': downsampled_grad,
            'downsampled_coordinate': downsampled_coordinate,
            }

