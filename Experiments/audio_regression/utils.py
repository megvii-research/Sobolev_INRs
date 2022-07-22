import os
import random

import numpy as np
import matplotlib.pyplot as plt

import torch


SAVE_FORMAT = "png"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_psnr(gt, pred, max_val=1.0):
    mse = (gt - pred).pow(2).mean()
    return 10.0 * torch.log10(max_val**2 / mse)


def write_train_summary(data, output, writer, epoch, out_train_imgs_dir):
    prefix = "train_"

    gt = data["downsampled_wav"]
    pred = output["pred"]

    # plot
    start_index = int(0.05 * len(gt))
    end_index = int(0.95 * len(gt))
    x_plot = (
        data["downsampled_coordinate"][start_index:end_index].detach().cpu().numpy()
    )
    gt_plot = gt[start_index:end_index].detach().cpu().numpy()
    pred_plot = pred[start_index:end_index].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(x_plot, gt_plot)
    axes[1].plot(x_plot, pred_plot)
    axes[2].plot(x_plot, gt_plot - pred_plot)
    axes[0].set_ylim([-0.75, 0.75])
    axes[1].set_ylim([-0.75, 0.75])
    axes[2].set_ylim([-0.25, 0.25])

    axes[0].set_ylabel("GT")
    axes[1].set_ylabel("Pred")
    axes[2].set_ylabel("Error")

    fig.savefig(
        os.path.join(out_train_imgs_dir, "%05d.%s" % (epoch, SAVE_FORMAT)),
        format=SAVE_FORMAT,
    )
    plt.close(fig)

    # write metric
    psnr = cal_psnr(gt, pred)
    writer.add_scalar(prefix + "PSNR", psnr, epoch)

    return psnr


def write_test_summary(
    data, output, writer, epoch, out_test_imgs_dir, factor, filename
):
    prefix = "test_"

    gt = data["wav"]
    pred = output["pred"]

    start_index = int(0.05 * len(gt))
    end_index = int(0.95 * len(gt))
    x_plot = data["coordinate"][start_index:end_index].detach().cpu().numpy()
    gt_plot = gt[start_index:end_index].detach().cpu().numpy()
    pred_plot = pred[start_index:end_index].detach().cpu().numpy()

    zoom_length = 600
    if "counting" in filename:
        zoom_start_index = 445000
    elif "bach" in filename:
        zoom_start_index = 150000
    else:
        raise NotImplementedError()
    zoom_end_index = zoom_start_index + zoom_length
    # print("zoom index: %d ~ %d" % (zoom_start_index, zoom_end_index))
    zoom_x_plot = (
        data["coordinate"][zoom_start_index:zoom_end_index].detach().cpu().numpy()
    )
    zoom_gt_plot = gt[zoom_start_index:zoom_end_index].detach().cpu().numpy()
    zoom_pred_plot = pred[zoom_start_index:zoom_end_index].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2)
    axes[0, 0].plot(x_plot, gt_plot)
    axes[1, 0].plot(x_plot, pred_plot)
    axes[2, 0].plot(x_plot, gt_plot - pred_plot)
    axes[0, 1].plot(zoom_x_plot, zoom_gt_plot)
    axes[1, 1].plot(zoom_x_plot, zoom_pred_plot)
    axes[2, 1].plot(zoom_x_plot, zoom_gt_plot - zoom_pred_plot)

    value_ylim = [-0.75, 0.75]
    diff_ylim = [-0.25, 0.25]
    axes[0, 0].set_ylim(value_ylim)
    axes[1, 0].set_ylim(value_ylim)
    axes[2, 0].set_ylim(diff_ylim)
    axes[0, 1].set_ylim(value_ylim)
    axes[1, 1].set_ylim(value_ylim)
    axes[2, 1].set_ylim(diff_ylim)

    axes[0, 0].axvline(x=zoom_start_index, color="r", linewidth=0.5)
    # axes[0, 0].axvline(x=zoom_end_index, color='r', linewidth=0.5)
    axes[1, 0].axvline(x=zoom_start_index, color="r", linewidth=0.5)
    axes[2, 0].axvline(x=zoom_start_index, color="r", linewidth=0.5)

    axes[0, 0].set_ylabel("GT")
    axes[1, 0].set_ylabel("Pred")
    axes[2, 0].set_ylabel("Error")

    fig.savefig(
        os.path.join(out_test_imgs_dir, "%05d.%s" % (epoch, SAVE_FORMAT)),
        format=SAVE_FORMAT,
    )
    plt.close(fig)

    # write metric
    eval_mask = torch.ones_like(gt, dtype=torch.bool)
    eval_mask[::factor, :] = False
    eval_gt = gt[eval_mask].clone()
    eval_pred = pred[eval_mask].clone()
    psnr = cal_psnr(eval_gt, eval_pred)
    writer.add_scalar(prefix + "PSNR", psnr, epoch)

    return psnr
