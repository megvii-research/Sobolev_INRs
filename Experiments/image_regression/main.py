import os
import shutil


import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import get_data
from model import MLP
from loss import *
from utils import *

set_random_seed(0)


def config_parser():

    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="Path of config file.")

    # logging options
    parser.add_argument(
        "--logging_root",
        type=str,
        default="./logs/",
        help="Where to store ckpts and logs.",
    )
    parser.add_argument(
        "--epochs_til_ckpt",
        type=int,
        default=1000,
        help="Time interval in epochs until checkpoint is saved.",
    )
    parser.add_argument(
        "--epochs_til_summary",
        type=int,
        default=100,
        help="Time interval in epochs until tensorboard summary is saved.",
    )

    # training options
    parser.add_argument("--lrate", type=float, default="1e-4")
    parser.add_argument(
        "--num_epochs", type=int, default=50000, help="Number of epochs to train for."
    )

    # experiment options
    parser.add_argument(
        "--exp_name",
        type=str,
        default="supervision_val_der",
        help="Name of experiment.",
    )
    parser.add_argument(
        "--supervision", type=str, default="val_der", choices=("val", "der", "val_der")
    )
    parser.add_argument(
        "--activations", nargs="+", default=["sine", "sine", "sine", "sine"]
    )
    parser.add_argument("--w0", type=float, default="30.")
    parser.add_argument("--is_gray", action="store_true")
    parser.add_argument(
        "--der_operator", type=str, default="sobel", choices=("sobel", "diff")
    )
    parser.add_argument("--has_pos_encoding", action="store_true")
    parser.add_argument("--has_fourier_feature", action="store_true")
    parser.add_argument("--lambda_der", type=float, default="1.")

    # model options
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=3)

    # dataset options
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../data",
        help="Root path to image datasets.",
    )
    parser.add_argument(
        "--dataset", type=str, default="Set5", choices=("Set5", "DIV2K_valid")
    )
    parser.add_argument("--filename", type=str, help="Name of image file.")
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        choices=(1, 2, 3, 4),
        help="Factor of downsampling.",
    )

    return parser


def train(
    args,
    model,
    data,
    epochs,
    lrate,
    epochs_til_summary,
    epochs_til_checkpoint,
    logging_dir,
    train_summary_fn,
    test_summary_fn,
    log_f,
):

    summaries_dir = os.path.join(logging_dir, "summaries")
    os.makedirs(summaries_dir)
    writer = SummaryWriter(summaries_dir)

    checkpoints_dir = os.path.join(logging_dir, "checkpoints")
    os.makedirs(checkpoints_dir)

    out_train_imgs_dir = os.path.join(logging_dir, "out_train_imgs")
    os.makedirs(out_train_imgs_dir)

    out_test_imgs_dir = os.path.join(logging_dir, "out_test_imgs")
    os.makedirs(out_test_imgs_dir)

    optim = torch.optim.Adam(lr=lrate, params=model.parameters())

    img_shape = data["img_shape"]
    downsampled_img_shape = data["downsampled_img_shape"]
    # move data to GPU
    data = {key: value.cuda() for key, value in data.items() if torch.is_tensor(value)}

    for epoch in range(1, epochs + 1):

        # forward and calculate loss
        model_output = model(data["downsampled_coordinate"], mode="train")
        losses = {}
        losses.update(val_mse(data["downsampled_img"], model_output["pred"]))
        losses.update(der_mse(data["downsampled_grad"], model_output["pred_grad"]))
        if args.supervision == "val":
            train_loss = losses["val_loss"]
        elif args.supervision == "der":
            train_loss = losses["der_loss"]
        elif args.supervision == "val_der":
            train_loss = 1.0 * losses["val_loss"] + args.lambda_der * losses["der_loss"]

        # tensorboard
        for loss_name, loss in losses.items():
            writer.add_scalar(loss_name, loss, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)

        # backward
        optim.zero_grad()
        train_loss.backward()
        optim.step()

        if (not epoch % epochs_til_summary) or (epoch == epochs):

            # training summary
            psnr, ssim = train_summary_fn(
                data,
                model_output,
                writer,
                epoch,
                downsampled_img_shape,
                out_train_imgs_dir,
            )

            str_print = "[Train] Epoch: (%d/%d) " % (epoch, epochs)
            for loss_name, loss in losses.items():
                str_print += loss_name + ": %0.6f, " % loss
            str_print += "PSNR: %.3f, SSIM: %.4f, " % (psnr, ssim)
            print(str_print)
            print(str_print, file=log_f)

            # test summary
            with torch.no_grad():
                model_output = model(data["coordinate"], mode="test")
                psnr, ssim = test_summary_fn(
                    data,
                    model_output,
                    writer,
                    epoch,
                    img_shape,
                    out_test_imgs_dir,
                    factor=args.factor,
                )
                str_print = "[Test]: PSNR: %.3f, SSIM: %.4f" % (psnr, ssim)
                print(str_print)
                print(str_print, file=log_f)

        # save checkpoint
        if (not epoch % epochs_til_checkpoint) or (epoch == epochs):
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, "model_epoch_%05d.pth" % epoch),
            )

    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "model_final.pth"))


def main():

    parser = config_parser()
    args = parser.parse_args()

    logging_dir = os.path.join(args.logging_root, args.exp_name)
    if os.path.exists(logging_dir):
        if (
            input("The logging directory %s exists. Overwrite? (y/n)" % logging_dir)
            == "y"
        ):
            shutil.rmtree(logging_dir)
    os.makedirs(logging_dir)

    with open(os.path.join(logging_dir, "log.txt"), "w") as log_f:

        print("Args:\n", args)
        print("Args:\n", args, file=log_f)

        data = get_data(
            args.data_root,
            args.dataset,
            args.filename,
            args.is_gray,
            args.factor,
            args.der_operator,
        )
        print("Shape of original image:", data["img_shape"])
        print("Shape of downsampled image:", data["downsampled_img_shape"])

        if args.is_gray:
            out_features = 1
        else:
            out_features = 3

        model = MLP(
            in_features=2,
            out_features=out_features,
            w0=args.w0,
            activations=args.activations,
            hidden_features=args.hidden_features,
            num_hidden_layers=args.num_hidden_layers,
            has_pos_encoding=args.has_pos_encoding,
            has_fourier_feature=args.has_fourier_feature,
            shape=data["img_shape"],
            sidelength=data["downsampled_img_shape"],
        )
        model.cuda()

        train(
            args=args,
            model=model,
            data=data,
            epochs=args.num_epochs,
            lrate=args.lrate,
            epochs_til_summary=args.epochs_til_summary,
            epochs_til_checkpoint=args.epochs_til_ckpt,
            logging_dir=logging_dir,
            train_summary_fn=write_train_summary,
            test_summary_fn=write_test_summary,
            log_f=log_f,
        )


if __name__ == "__main__":
    main()
