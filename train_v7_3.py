# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

# 这个版本为加入相机编码版本（评估稳定化：rank0-only + DreamSim 单例）

import torch
import torch.nn as nn

# 让 A100 更快（按你原先设置保留）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import matplotlib.pyplot as plt
import yaml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from diffusers.models import AutoencoderKL

from isolated_nwm_infer_v7_2 import model_forward_wrapper

from distributed import init_distributed
from models_tpz_v7_2 import CDiT_models
from diffusion import create_diffusion
from datasets_v3 import TrainingDataset
from misc import transform

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#################################################################################
#                          DreamSim 单例（仅 rank0 构建一次）                   #
#################################################################################

_DREAMSIM_SINGLETON = None

def get_dreamsim(device):
    global _DREAMSIM_SINGLETON
    if _DREAMSIM_SINGLETON is None:
        # 仅在 rank0 会被调用（我们训练循环里会做 rank 判断）
        from dreamsim import dreamsim
        model, _ = dreamsim(pretrained=True)
        model = model.to(device).eval()
        _DREAMSIM_SINGLETON = model
    return _DREAMSIM_SINGLETON

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    _, rank, device, _ = init_distributed()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # Setup dirs:
    os.makedirs(config['results_dir'], exist_ok=True)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model & VAE:
    tokenizer = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config['image_size'] // 8
    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    num_cond = config['context_size']
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # Optimizer:
    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    bfloat_enable = bool(getattr(args, 'bfloat16', 0))
    scaler = torch.amp.GradScaler('cuda') if bfloat_enable else None

    # Resume:
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print('Searching for model from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0
    if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
        if os.path.isfile(latest_path) and config.get('from_checkpoint', 0):
            raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
        latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False)

        if "model" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['model'].items()}
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            ema_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['ema'].items()}
            res = ema.load_state_dict(ema_ckp, strict=True)
            print("Loading EMA model weights", res)
        else:
            update_ema(ema, model, decay=0)

        if "opt" in latest_checkpoint:
            opt_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['opt'].items()}
            opt.load_state_dict(opt_ckp)
            print("Loading optimizer params")

        if "epoch" in latest_checkpoint:
            start_epoch = latest_checkpoint['epoch'] + 1

        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]

        if "scaler" in latest_checkpoint and scaler is not None:
            scaler.load_state_dict(latest_checkpoint["scaler"])

    # Compile（可选）
    if getattr(args, 'torch_compile', 0):
        model = torch.compile(model)
    model = DDP(model, device_ids=[device])

    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets:
    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        for split in ["train", "test"]:
            if split in data_config:
                goals_per_obs = int(data_config["goals_per_obs"])
                if split == 'test':
                    goals_per_obs = 4
                if "distance" in data_config:
                    min_dist_cat = data_config["distance"]["min_dist_cat"]
                    max_dist_cat = data_config["distance"]["max_dist_cat"]
                else:
                    min_dist_cat = config["distance"]["min_dist_cat"]
                    max_dist_cat = config["distance"]["max_dist_cat"]

                len_traj_pred = data_config.get("len_traj_pred", config["len_traj_pred"])

                dataset = TrainingDataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[split],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    min_dist_cat=min_dist_cat,
                    max_dist_cat=max_dist_cat,
                    len_traj_pred=len_traj_pred,
                    context_size=config["context_size"],
                    normalize=config["normalize"],
                    goals_per_obs=goals_per_obs,
                    transform=transform,
                    predefined_index=None,
                    traj_stride=1,
                )
                if split == "train":
                    train_dataset.append(dataset)
                else:
                    test_dataset.append(dataset)
                print(f"Dataset: {dataset_name} ({split}), size: {len(dataset)}")

    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Train:
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y, rel_t, aug, camera_mats in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            aug = aug.to(device, non_blocking=True)                   # [B, num_goals, 4, 28, 28]
            camera_mats = camera_mats.to(device, non_blocking=True)   # [B, num_goals+num_cond, 4, 4]

            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Encode to latents:
                    B, T = x.shape[:2]
                    x = x.flatten(0, 1)
                    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215)
                    x = x.unflatten(0, (B, T))  # [B, num_goals+num_conds, 4, H', W']

                    B_aug, T_aug = aug.shape[:2]
                    aug = aug.flatten(0, 1)
                    aug = tokenizer.encode(aug).latent_dist.sample().mul_(0.18215)
                    aug = aug.unflatten(0, (B_aug, T_aug))  # [B, num_goals, 4, H', W']

                num_goals = T - num_cond
                x_start = x[:, num_cond:].flatten(0, 1)  # [B*num_goals, 4, H', W']
                x_cond = (
                    x[:, :num_cond]
                    .unsqueeze(1)
                    .expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4])
                    .flatten(0, 1)
                )  # [B*num_goals, num_cond, 4, H', W']
                y_cond = aug.unsqueeze(2).flatten(0, 1)  # [B*num_goals, 1, 4, H', W']

                y = y.flatten(0, 1)
                rel_t = rel_t.flatten(0, 1)

                camera_mats_x_start = camera_mats[:, num_cond:].unsqueeze(2).flatten(0, 1)  # [B*num_goals, 1, 4, 4]
                camera_mats_x_cond = (
                    camera_mats[:, :num_cond]
                    .unsqueeze(1)
                    .expand(B, num_goals, num_cond, 4, 4)
                    .flatten(0, 1)
                )  # [B*num_goals, num_cond, 4, 4]
                camera_mats_x_cond = torch.cat((camera_mats_x_cond, camera_mats_x_start), dim=1)  # [B*num_goals, num_conds+1, 4, 4]

                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)
                model_kwargs = dict(
                    y=y,
                    x_cond=x_cond,
                    rel_t=rel_t,
                    x_sup=y_cond.squeeze(1),
                    viewmats=camera_mats_x_cond,
                )
                loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            if not bfloat_enable:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                if config.get('grad_clip_val', 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                scaler.step(opt)
                scaler.update()

            update_ema(ema, model.module)

            # Logging:
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = dist.get_world_size() * x_cond.shape[0] * steps_per_sec
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}"
                )
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    if bfloat_enable:
                        checkpoint.update({"scaler": scaler.state_dict()})
                    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                    torch.save(checkpoint, checkpoint_path)
                    if train_steps % (10 * args.ckpt_every) == 0:
                        torch.save(checkpoint, f"{checkpoint_dir}/{train_steps:07d}.pth.tar")
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Eval (rank0-only + DreamSim 单例)
            if train_steps % args.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                if rank == 0:
                    eval_model = get_dreamsim(device)  # 单例
                    sim_score = evaluate(
                        ema, tokenizer, diffusion, test_dataset, rank,
                        config["batch_size"], config["num_workers"], latent_size,
                        device, save_dir, args.global_seed, bfloat_enable, num_cond,
                        eval_model
                    )
                    eval_end_time = time()
                    logger.info(f"(step={train_steps:07d}) Perceptual Loss: {float(sim_score):.4f}, Eval Time: {eval_end_time - eval_start_time:.2f}")
                # 其它 rank 等 rank0 评估完
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


@torch.no_grad()
def evaluate(model, vae, diffusion, test_dataloaders, rank, batch_size, num_workers,
             latent_size, device, save_dir, seed, bfloat_enable, num_cond, eval_model):
    """
    仅 rank0 调用；eval_model 为 DreamSim 单例（已在 eval 前 .to(device).eval()）
    """
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )
    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,  # eval 不需要常驻 worker
    )

    score = torch.tensor(0., device=device)
    n_samples = torch.tensor(0, device=device)

    # 只跑一个 batch
    for x, y, rel_t, aug, camera_mats in loader:
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        aug = aug.to(device)
        camera_mats = camera_mats.to(device)

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            samples = model_forward_wrapper(
                (model, diffusion, vae),
                x, y, num_timesteps=None, latent_size=latent_size, device=device,
                num_cond=num_cond, num_goals=num_goals, rel_t=rel_t, x_supervised=aug, camera_mats=camera_mats
            )
            x_start_pixels = x[:, num_cond:].flatten(0, 1)
            x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)

            # 反归一化到 [0,1]
            samples = samples * 0.5 + 0.5
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            x_cond_pixels = x_cond_pixels * 0.5 + 0.5

            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)
        break

    # 保存若干可视化
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        K = min(samples.shape[0], 10)
        for i in range(K):
            fig, ax = plt.subplots(1, 3, dpi=256)
            ax[0].imshow((x_cond_pixels[i, -1].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8'))
            ax[0].set_title('cond[-1]')
            ax[0].axis('off')
            ax[1].imshow((x_start_pixels[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8'))
            ax[1].set_title('gt')
            ax[1].axis('off')
            ax[2].imshow((samples[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8'))
            ax[2].set_title('pred')
            ax[2].axis('off')
            fig.tight_layout()
            fig.savefig(f'{save_dir}/{i}.png')
            plt.close(fig)

    # 在本函数中 rank==0 才会进来，这两行只是保持接口一致（不影响结果）
    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score / n_samples.clamp_min(1)
    return sim_score


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
