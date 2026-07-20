# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import os
import numpy as np

from nwm.config import load_runtime_config, split_path
from nwm.data.airvln import EvalDataset
from nwm.diffusion import create_diffusion
from nwm.distributed import init_distributed
import nwm.distributed as dist
from nwm.model import CDiT_models
from nwm.rollout import model_forward_wrapper
import nwm.utils as misc
from PIL import Image


def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = misc.unnormalize(img)

    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode="RGB")

    image.save(output_file)


def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]
    if predefined_index:
        predefined_index = str(split_path(dataset_name, f"{eval_type}_16.pkl"))
        # predefined_index = f"data_splits/airvln_16/test/dataset_dist_-64_to_64_n4_len_traj_pred_64.pkl"
    else:
        predefined_index = None

    dataset = EvalDataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
        max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
        len_traj_pred=config["eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["eval_context_size"],
        normalize=config["normalize"],
        transform=misc.transform,
        goals_per_obs=4,
        predefined_index=predefined_index,
        traj_names="traj_names.txt",
    )

    return dataset


def generate_rollout(
    args,
    output_dir,
    rollout_fps,
    idxs,
    all_models,
    obs_image,
    gt_image,
    delta,
    num_cond,
    device,
    x_supervised,
):
    rollout_stride = args.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride - 1 :: rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
    curr_obs = obs_image.clone().to(device)
    sup_image = x_supervised[
        :, rollout_stride - 1 :: rollout_stride
    ]  # (B, T_roll, C, H, W)
    assert (
        sup_image.shape == gt_image.shape
    ), f"x_sup_strided shape={sup_image.shape} != gt shape={gt_image.shape}"

    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i : i + 1].to(device)
        if args.gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                rollout_stride,
                args.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                x_supervised=sup_image[:, i : i + 1],
            )

        curr_obs = torch.cat(
            (curr_obs, x_pred_pixels.unsqueeze(1)), dim=1
        )  # append current prediction
        curr_obs = curr_obs[:, 1:]  # remove first observation
        visualize_preds(output_dir, idxs, i, x_pred_pixels)


def generate_time_original(
    args,
    output_dir,
    idxs,
    all_models,
    obs_image,
    gt_output,
    delta,
    secs,
    num_cond,
    device,
    x_supervised,
):
    eval_timesteps = [sec * args.input_fps for sec in secs]
    for sec, timestep in zip(secs, eval_timesteps):
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)
        if args.gt:
            x_pred_pixels = gt_output[:, timestep - 1].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                obs_image,
                curr_delta,
                timestep,
                args.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                x_supervised=x_supervised,
            )
        visualize_preds(output_dir, idxs, sec, x_pred_pixels)


def generate_time(
    args,
    output_dir,
    idxs,
    all_models,
    obs_image,
    gt_output,
    delta,
    secs,
    num_cond,
    device,
    x_supervised,
):
    eval_timesteps = [sec * args.input_fps for sec in secs]
    num_goals = len(secs)
    B = obs_image.size(0)

    # 1. 构造多个 goal 的动作输入（B, num_goals, 4)
    delta_goals = []
    for timestep in eval_timesteps:
        d = delta[:, :timestep].sum(dim=1, keepdim=True)  # (B, 1, 4)
        delta_goals.append(d)
    delta_goals = torch.cat(delta_goals, dim=1)  # (B, num_goals, 4)

    # 2. 扩展所有输入以匹配 B * num_goals
    # obs_image = obs_image.unsqueeze(1).repeat(1, num_goals, 1, 1, 1, 1).flatten(0, 1)  # (B*num_goals, T, C, H, W)
    delta_goals = delta_goals.flatten(0, 1)  # (B*num_goals, 4)

    if x_supervised is not None:
        x_supervised = (
            x_supervised.unsqueeze(1).repeat(1, num_goals, 1, 1, 1, 1).flatten(0, 1)
        )  # (B*num_goals, T, C, H, W)

    if args.gt:
        # 直接从 ground truth 中提取目标帧
        x_pred_pixels = []
        for timestep in eval_timesteps:
            x_pred_pixels.append(
                gt_output[:, timestep - 1].clone().to(device)
            )  # (B, C, H, W)
        x_pred_pixels = torch.stack(x_pred_pixels, dim=1)  # (B, num_goals, C, H, W)
    else:
        # 3. 执行一次 forward，预测多个目标时间点
        x_pred_pixels = model_forward_wrapper(
            all_models,
            obs_image,
            delta_goals,
            max(eval_timesteps),
            args.latent_size,
            num_cond=num_cond,
            num_goals=num_goals,
            device=device,
            x_supervised=x_supervised,
        )  # 返回 (B * num_goals, C, H, W)
        x_pred_pixels = x_pred_pixels.view(
            B, num_goals, *x_pred_pixels.shape[1:]
        )  # (B, num_goals, C, H, W)

    for goal_idx, sec in enumerate(secs):
        visualize_preds(output_dir, idxs, sec, x_pred_pixels[:, goal_idx])


def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs.squeeze()):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f"id_{sample_idx}")
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f"{sec}.png")
        save_image(image_file, x_pred_pixels[batch_idx], True)


@torch.no_grad()
def main(args):
    _, _, device, _ = init_distributed()
    print(args)
    device = torch.device(device)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    exp_eval = args.exp

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, "gt")
    else:
        exp_name = os.path.basename(exp_eval).split(".")[0]
        args.save_output_dir = os.path.join(args.output_dir, exp_name)

    if not args.gt and args.ckp != "0200000":
        args.save_output_dir = args.save_output_dir + "_%s" % (args.ckp)

    os.makedirs(args.save_output_dir, exist_ok=True)

    config = load_runtime_config(exp_eval)

    latent_size = config["image_size"] // 8
    args.latent_size = config["image_size"] // 8

    num_cond = config["context_size"]
    print("loading")
    model_lst = (None, None, None)
    if not args.gt:
        from diffusers.models import AutoencoderKL

        model = CDiT_models[config["model"]](
            context_size=num_cond, input_size=latent_size, in_channels=4
        )
        ckp = torch.load(
            f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar',
            map_location="cpu",
            weights_only=False,
        )
        print(model.load_state_dict(ckp["ema"], strict=True))
        model.eval()
        model.to(device)
        model = torch.compile(model)
        diffusion = create_diffusion(str(250))
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=False
        )
        model_lst = (model, diffusion, vae)

    # Loading Datasets
    dataset_names = args.datasets.split(",")
    datasets = {}

    for dataset_name in dataset_names:
        dataset_val = get_dataset_eval(
            config, dataset_name, args.eval_type, predefined_index=True
        )
        # dataset_val = Subset(dataset_val, list(range(1000)))
        # print(len(dataset_val))
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

        curr_data_loader = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        datasets[dataset_name] = curr_data_loader

    print_freq = 1
    header = "Evaluation: "
    metric_logger = dist.MetricLogger(delimiter="  ")

    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]

        for data_iter_step, (idxs, obs_image, gt_image, delta, aug_image) in enumerate(
            metric_logger.log_every(curr_data_loader, print_freq, header)
        ):
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                obs_image = obs_image[:, -num_cond:].to(device)
                gt_image = gt_image.to(device)
                num_cond = config["context_size"]
                if args.eval_type == "rollout":
                    for rollout_fps in args.rollout_fps_values:
                        curr_rollout_output_dir = os.path.join(
                            dataset_save_output_dir, f"rollout_{rollout_fps}fps"
                        )
                        os.makedirs(curr_rollout_output_dir, exist_ok=True)
                        generate_rollout(
                            args,
                            curr_rollout_output_dir,
                            rollout_fps,
                            idxs,
                            model_lst,
                            obs_image,
                            gt_image,
                            delta,
                            num_cond,
                            device,
                            aug_image,
                        )
                elif args.eval_type == "time":
                    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
                    curr_time_output_dir = os.path.join(dataset_save_output_dir, "time")
                    os.makedirs(curr_time_output_dir, exist_ok=True)
                    generate_time(
                        args,
                        curr_time_output_dir,
                        idxs,
                        model_lst,
                        obs_image,
                        gt_image,
                        delta,
                        secs,
                        num_cond,
                        device,
                        aug_image,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir", type=str, default="outputs/inference", help="output directory"
    )
    parser.add_argument(
        "--exp", type=str, default="config/v5_3.yaml", help="experiment config"
    )
    parser.add_argument("--ckp", type=str, default="0200000")
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument(
        "--datasets", type=str, default="airvln_16", help="dataset name"
    )
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--eval_type", type=str, default="rollout", choices=["time", "rollout"]
    )
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default="1,4", help="")
    parser.add_argument(
        "--gt",
        type=int,
        default=0,
        help="set to 1 to produce ground truth evaluation set",
    )
    args = parser.parse_args()

    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(",")]

    main(args)
