import os
import time

import imageio
import kornia
import einops
import numpy as np
from tqdm import tqdm, trange

import torch

from datasets.load_llff import load_llff_data
from datasets.ray_utils import get_rays
from models.nerf import TinyNeRF, get_embedder
from models.rendering import render, render_path
from loss import img2mse, mse2psnr, to8b, der_mse
from utils import set_random_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(seed=0)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def create_tiny_nerf(args):
    """Instantiate TinyNeRF model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4
    model = TinyNeRF(
        D=args.netdepth,
        activations=args.activations,
        W=args.netwidth,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
        output_ch=output_ch,
    ).to(device)
    print("model: ", model)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])

    ##########################

    model = torch.nn.DataParallel(model)

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():

    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="../../data/nerf_llff_data/fern",
        help="input data directory",
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--N_rand",
        type=int,
        default=1024,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=128, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=1e0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    ## llff flags
    parser.add_argument(
        "--llff_factor", type=int, default=4, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=50000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=400000,
        help="frequency of render_poses video saving",
    )

    # added options
    parser.add_argument("--activations", nargs="+", default=["sine"] * 8)
    parser.add_argument(
        "--der_operator", type=str, default="sobel", choices=("sobel", "diff")
    )
    parser.add_argument(
        "--supervision", type=str, default="val_der", choices=("val", "der", "val_der")
    )
    parser.add_argument("--lambda_der", type=float, default=1.0)
    parser.add_argument(
        "--show_der_loss",
        action="store_true",
        help="for supervision=val, show der_loss values",
    )

    # downsample options
    parser.add_argument("--factor", type=int, default=4)

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    print("args: ", args)

    # Load LLFF data
    images, poses, bds, render_poses, i_test = load_llff_data(
        args.datadir,
        args.llff_factor,
        recenter=True,
        bd_factor=0.75,
        spherify=args.spherify,
    )
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print("Auto LLFF holdout,", args.llffhold)
        i_test = np.arange(images.shape[0])[:: args.llffhold]

    i_val = i_test
    i_train = np.array(
        [
            i
            for i in np.arange(int(images.shape[0]))
            if (i not in i_test and i not in i_val)
        ]
    )

    print("DEFINING BOUNDS")
    if args.no_ndc:
        near = np.ndarray.min(bds) * 0.9
        far = np.ndarray.max(bds) * 1.0

    else:
        near = 0.0
        far = 1.0
    print("NEAR FAR", near, far)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    # calculate first-order image derivatives
    grad = kornia.filters.spatial_gradient(
        einops.rearrange(torch.Tensor(images), "b h w c -> b c h w"),
        mode=args.der_operator,
        order=1,
        normalized=True,
    )  # [B, C, 2, H, W], grad_x = grad[:, 0], grad_y = grad[:, 1]
    grad = einops.rearrange(grad, "b c d h w -> b h w (c d)")  # [B, H, W, C*2]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    train_logfile = os.path.join(basedir, expname, "train_log.txt")

    # Create nerf model
    (
        render_kwargs_train,
        render_kwargs_test,
        start,
        grad_vars,
        optimizer,
    ) = create_tiny_nerf(args)
    global_step = start

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print("RENDER ONLY")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                basedir,
                expname,
                "renderonly_{}_{:06d}".format(
                    "test" if args.render_test else "path", start
                ),
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)

            rgbs, _ = render_path(
                render_poses,
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=args.render_factor,
            )
            print("Done rendering", testsavedir)
            imageio.mimwrite(
                os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8
            )

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # For random ray batching
    images = torch.Tensor(images).to(device)  # [N, H, W, 3]
    print("get rays")
    coordinate = torch.stack(
        torch.meshgrid(
            torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="xy"
        ),
        -1,
    )  # [H, W, 2]
    coordinate = coordinate.unsqueeze(0).expand(
        images.shape[0], -1, -1, -1
    )  # [N, H, W, 2]

    downsampled_images = images[
        :, :: args.factor, :: args.factor, :
    ]  # [N, H//factor, W//factor, 3]
    print("Downsampled images shape: ", downsampled_images.shape)
    downsampled_coordinate = coordinate[
        :, :: args.factor, :: args.factor, :
    ]  # [N, H//factor, W//factor, 2]
    downsampled_grad = grad[
        :, :: args.factor, :: args.factor, :
    ]  # [N, H//factor, W//factor, 3*2]

    poses = torch.Tensor(poses).to(device)  # [N, 3, 4]
    data = torch.cat(
        [
            downsampled_images,
            downsampled_coordinate,
            einops.rearrange(poses, "n e f -> n () () (e f)").expand(
                -1, H // args.factor, W // args.factor, -1
            ),
            downsampled_grad,
        ],
        dim=-1,
    )  # [B, H//factor, W//factor, 3+2+12+c*2], (rgb, xy, pose, grad)
    data = torch.stack(
        [data[i] for i in i_train]
    )  # [N_train, H//factor, W//factor, 3+2+12+3*2]
    data = einops.rearrange(
        data, "n h w c -> (n h w) c"
    )  # [N_train*(H//factor)*(W//factor), 3+2+12+3*2], (rgb, xy, pose, grad)

    print("shuffle data")
    rand_idx = torch.randperm(data.shape[0])
    data = data[rand_idx]

    print("done")
    i_batch = 0

    N_iters = 400000 + 1
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch, random over all images
        batch_data = data[
            i_batch : i_batch + N_rand
        ]  # [N_train*H*W, 3+2+12+c*2], (rgb, xy, pose, grad)
        target_s = batch_data[:, :3]
        coordinate_s = (
            batch_data[:, 3:5].float().clone().detach().requires_grad_(True)
        )  # enable us to compute gradients w.r.t. coordinates(xy)
        poses_s = batch_data[:, 5:17]
        poses_s = einops.rearrange(poses_s, "b (e f) -> b e f", e=3)  # [B, 12]
        target_grad_s = batch_data[:, 17:]  # [B, c*2]
        rays_o, rays_d = get_rays(K, poses_s, coordinate_s)
        batch_rays = torch.stack([rays_o, rays_d], dim=0)  # [2, B, 3]

        i_batch += N_rand
        if i_batch >= data.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(data.shape[0])
            data = data[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(
            H,
            W,
            K,
            chunk=args.chunk,
            rays=batch_rays,
            verbose=i < 10,
            retraw=True,
            **render_kwargs_train,
        )

        optimizer.zero_grad()
        val_loss = img2mse(rgb, target_s)
        trans = extras["raw"][..., -1]

        if args.supervision == "val":
            loss = val_loss
            if args.show_der_loss:
                der_loss = der_mse(rgb, coordinate_s, target_grad_s)
                der_loss = der_loss.item()
        elif args.supervision == "der":
            der_loss = der_mse(rgb, coordinate_s, target_grad_s)
            loss = der_loss
        else:  # args.supervision == 'val_der'
            der_loss = der_mse(rgb, coordinate_s, target_grad_s)
            loss = 1.0 * val_loss + args.lambda_der * der_loss

        psnr = mse2psnr(val_loss)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoints at", path)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                render_path(
                    torch.Tensor(poses[i_test]).to(device),
                    hwf,
                    K,
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=images[i_test],
                    savedir=testsavedir,
                )
            print("Saved test set")

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses, hwf, K, args.chunk, render_kwargs_test
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, "spiral_{:06d}_".format(i))
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8
            )

        if i % args.i_print == 0:
            if args.supervision == "val":
                if args.show_der_loss:
                    out_str = f"[TRAIN] Iter: {i}  val_loss: {val_loss.item()}  der_loss: {der_loss.item()}  PSNR: {psnr.item()}"
                else:
                    out_str = f"[TRAIN] Iter: {i}  val_loss: {val_loss.item()}  PSNR: {psnr.item()}"
            else:
                out_str = f"[TRAIN] Iter: {i}  val_loss: {val_loss.item()}  der_loss: {der_loss.item()}  PSNR: {psnr.item()}"
            tqdm.write(out_str)
            with open(train_logfile, "a") as file:
                file.write(out_str + "\n")

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    train()
