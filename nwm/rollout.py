"""Shared image rollout used by training, inference, and planning."""

import torch


@torch.no_grad()
def model_forward_wrapper(
    all_models,
    curr_obs,
    curr_delta,
    num_timesteps,
    latent_size,
    device,
    num_cond,
    num_goals=1,
    rel_t=None,
    progress=False,
    x_supervised=None,
):
    model, diffusion, vae = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)
    if x_supervised is None:
        raise ValueError("x_supervised is required for the v5_3 model")
    x_supervised = x_supervised.to(device)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        batch_size, num_frames = x.shape[:2]
        if rel_t is None:
            if num_timesteps is None:
                raise ValueError("num_timesteps is required when rel_t is not provided")
            rel_t = torch.full(
                (batch_size * num_goals,),
                num_timesteps / 128.0,
                device=device,
            )

        x = x.flatten(0, 1)
        x = (
            vae.encode(x)
            .latent_dist.sample()
            .mul_(0.18215)
            .unflatten(0, (batch_size, num_frames))
        )
        x_cond = (
            x[:, :num_cond]
            .unsqueeze(1)
            .expand(batch_size, num_goals, num_cond, *x.shape[2:])
            .flatten(0, 1)
        )
        latent = torch.randn(
            batch_size * num_goals,
            4,
            latent_size,
            latent_size,
            device=device,
        )

        y = y.flatten(0, 1)
        supervised_batch, supervised_frames = x_supervised.shape[:2]
        supervised = x_supervised.flatten(0, 1)
        supervised = (
            vae.encode(supervised)
            .latent_dist.sample()
            .mul_(0.18215)
            .unflatten(0, (supervised_batch, supervised_frames))
            .flatten(0, 1)
        )

        model_kwargs = {
            "y": y,
            "x_cond": x_cond,
            "rel_t": rel_t,
            "x_supervised": supervised,
        }
        samples = diffusion.p_sample_loop(
            model.forward,
            latent.shape,
            latent,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=device,
        )
        samples = vae.decode(samples / 0.18215).sample
        return torch.clip(samples, -1.0, 1.0)
