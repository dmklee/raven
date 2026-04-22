from pathlib import Path


from PIL import Image
import os
from omegaconf import OmegaConf
import copy
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import random

from uferl.model.common.lr_scheduler import get_scheduler
from uferl.model.common.rotation_transformer import RotationTransformer
from uferl.common.pytorch_util import dict_apply


max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'two_arm_drawer_cleanup': 550,
    'two_arm_box_cleanup': 400,
    'two_arm_lift_tray': 750,
    'two_arm_threading': 400,
    'two_arm_transport': 1200,
    'two_arm_three_piece_assembly': 300,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
}

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)


@hydra.main(config_path="./config", config_name="default", version_base="1.1")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)

    # set seed
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    # get dataset/loader
    train_dataset = hydra.utils.instantiate(cfg.data)

    train_loader = hydra.utils.instantiate(
        cfg.dataloader, dataset=train_dataset
    )

    if cfg.training.debug:
        cfg.use_wandb = False

    # create env for evaluation
    if cfg.training.debug == False and cfg.training.eval_interval < cfg.training.num_epochs:
        env_runner = instantiate(
            cfg.task.env_runner, output_dir=cfg.output_dir
        )
    else:
        env_runner = None

    # create logger
    if cfg.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )

    # create policy
    device = torch.device(cfg.training.device)
    policy = instantiate(cfg.policy)
    normalizer = train_dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)

    if 'ema' in cfg:
        ema_policy = copy.deepcopy(policy).to(device)
        ema = instantiate(cfg.ema, model=ema_policy)
    else:
        ema = None

    optimizer = instantiate(
        cfg.optimizer, params=policy.parameters()
    )

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=(
            len(train_loader) * cfg.training.num_epochs
        ) // cfg.training.gradient_accumulate_every,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        last_epoch=-1 # update this if resuming
    )

    global_step = 0
    for epoch_id in tqdm(range(cfg.training.num_epochs), leave=True, desc="Epochs"):
        batch_id = 0

        train_sampling_batch = None
        for batch in tqdm(train_loader, leave=False, mininterval=1, desc="Batches"):
            ## update step
            if train_sampling_batch is None:
                train_sampling_batch = batch
                continue

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            raw_loss = policy.compute_loss(batch)
            loss = raw_loss / cfg.training.gradient_accumulate_every
            loss.backward()

            if 'clip_grad_norm' in cfg.training:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    cfg.training.clip_grad_norm
                )

            if global_step % cfg.training.gradient_accumulate_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if ema is not None:
                ema.step(policy)

            if cfg.use_wandb:
                raw_loss_cpu = raw_loss.item()
                wandb.log(
                    {
                        'train/loss' : raw_loss.item(),
                        'global_step' : global_step,
                        'epoch' : epoch_id,
                        'lr' : lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
            global_step += 1
            batch_id += 1

            if cfg.training.debug and batch_id >= 1:
                break

        # perform eval
        eval_policy = policy if ema is None else ema_policy
        eval_policy.eval()
        if (
            (
                epoch_id % cfg.training.eval_interval == 0
                or epoch_id == cfg.training.num_epochs - 1
            ) and epoch_id > 0 and env_runner is not None
        ):
            eval_data = env_runner.run(eval_policy)
            for k in list(eval_data.keys()):
                if 'sim_max_reward' in k:
                    del eval_data[k]

            if cfg.use_wandb:
                wandb.log(
                    eval_data,
                    step=global_step,
                )
            del eval_data

        if (epoch_id % cfg.training.sample_interval) == 0:
            with torch.no_grad():
                batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = batch['obs']
                gt_action = batch['action']
                B, T, Na = gt_action.shape
                gt_action = gt_action.view(B, T, cfg.task.n_arms, -1)

                eval_policy.reset()
                result = None
                pred_action = eval_policy.predict_action(obs_dict)['action_pred']

                pred_action = pred_action.view(B, T, cfg.task.n_arms, -1)

                mse = F.mse_loss(pred_action, gt_action)
                pos_err = torch.linalg.norm(
                    torch.flatten(pred_action[..., :3]-gt_action[..., :3], 0, 2), dim=1
                ).mean()
                rot_tfm = RotationTransformer(
                    'rotation_6d' if cfg.task.abs_action else 'axis_angle', 'matrix'
                )
                rot_dim = 6 if cfg.task.abs_action else 3
                pred_rotmats = rot_tfm.forward(pred_action[..., 3:3+rot_dim])
                gt_rotmats = rot_tfm.forward(gt_action[..., 3:3+rot_dim])
                prod = torch.matmul(pred_rotmats, gt_rotmats.transpose(-1, -2))
                trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
                rot_err = torch.arccos(torch.clamp((trace-1)/2, -1, 1)).mean()
                gripper_err = F.l1_loss(pred_action[..., 3+rot_dim:], gt_action[..., 3+rot_dim:])

                if cfg.use_wandb:
                    wandb.log(
                        {
                            'train/action_mse': mse.item(),
                            'train/pos_mse': pos_err.item(),
                            'train/rot_err': rot_err.item(),
                            'train/gripper_err': gripper_err.item(),
                        },
                        step=global_step,
                    )
                del batch
                del obs_dict
                del gt_action
                del result
                del pred_action
                del mse
                del pos_err
                del rot_err
                del gripper_err

        eval_policy.train()

        # save checkpoint
        if (
            (
                epoch_id % cfg.training.save_interval == 0
                or epoch_id == cfg.training.num_epochs - 1
            )
            and epoch_id > 0
        ):
            # print('Checkpointing not implemented')
            save_path = None


if __name__ == "__main__":
    main()
