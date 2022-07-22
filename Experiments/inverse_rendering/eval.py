import os
import glob

import imageio
import einops
import numpy as np

from decimal import Decimal

rounding_half_up_4 = lambda x: Decimal(str(x)).quantize(
    Decimal("0.0001"), rounding="ROUND_HALF_UP"
)

import torch

from datasets.load_llff import load_llff_data
from train import config_parser
from utils import cal_psnr, cal_ssim


def read_data(args):

    # Load LLFF data
    images, poses, bds, render_poses, i_test = load_llff_data(
        args.datadir, args.factor, recenter=True, bd_factor=0.75, spherify=args.spherify
    )
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print("Auto LLFF holdout,", args.llffhold)
        i_test = np.arange(images.shape[0])[:: args.llffhold]

    return images, i_test


def eval_img(gt, pred):
    gt = einops.rearrange(torch.tensor(gt).unsqueeze(0), "b h w c -> b c h w")
    pred = einops.rearrange(torch.tensor(pred).unsqueeze(0), "b h w c -> b c h w")

    return cal_psnr(gt, pred).item(), cal_ssim(gt, pred).item()


def eval_testset(testset_gt, testset_pred, f=None):
    psnr_list = []
    ssim_list = []
    for i, (gt, pred) in enumerate(zip(testset_gt, testset_pred)):
        psnr, ssim = eval_img(gt, pred)
        print(
            "%03d.png PSNR: %s, SSIM: %s"
            % (i, rounding_half_up_4(psnr), rounding_half_up_4(ssim))
        )
        print(
            "%03d.png PSNR: %s, SSIM: %s"
            % (i, rounding_half_up_4(psnr), rounding_half_up_4(ssim)),
            file=f,
        )
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    print(
        "Mean PSNR: %s, Mean SSIM: %s"
        % (
            rounding_half_up_4(np.mean(psnr_list)),
            rounding_half_up_4(np.mean(ssim_list)),
        )
    )
    print(
        "Mean PSNR: %s, Mean SSIM: %s"
        % (
            rounding_half_up_4(np.mean(psnr_list)),
            rounding_half_up_4(np.mean(ssim_list)),
        ),
        file=f,
    )


def eval_exp():
    parser = config_parser()
    args = parser.parse_args()
    print("args: ", args)

    exp_log_dir = os.path.join(args.basedir, args.expname)
    testset_pred_dirs = glob.glob(os.path.join(exp_log_dir, "testset_*"))
    if len(testset_pred_dirs) == 0:
        print("==============> EXP: %s, no testset result, skipping..." % args.expname)
        return

    print("==============> EXP: %s" % args.expname)

    testset_pred_to_be_processed = []
    for testset_pred_dir in testset_pred_dirs:
        score_file = os.path.join(testset_pred_dir, "score.txt")
        if os.path.exists(score_file):
            print("Evaluating: %s, score.txt exists, skipping..." % (testset_pred_dir))
        else:
            testset_pred_to_be_processed.append(testset_pred_dir)
    if len(testset_pred_to_be_processed) == 0:
        return

    images, i_test = read_data(args)

    for testset_pred_dir in testset_pred_to_be_processed:
        testset_pred = [
            imageio.imread(fname).astype(np.float32) / 255.0
            for fname in glob.glob(os.path.join(testset_pred_dir, "*.png"))
        ]
        if len(testset_pred) != len(i_test):
            print(
                "Evaluating: %s, the number of predicted images is wrong, skipping..."
                % (testset_pred_dir)
            )
        else:
            score_file = os.path.join(testset_pred_dir, "score.txt")
            with open(score_file, "w") as f:
                print("Evaluating: ", testset_pred_dir)
                eval_testset(images[i_test], testset_pred, f)


if __name__ == "__main__":
    eval_exp()
