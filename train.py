import argparse
import gym
import numpy as np
import os
import wandb

import utils
from datasets import SequenceDataset
from dc.dc import DiffuserCritic

class IterParser(utils.HparamEnv):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'
    experiment: str = 'diffusion'

iterparser = IterParser()

class Parser(utils.Parser):
    pid: int = 0
    cid: float = 0

args = Parser().parse_args(iterparser)

dataset = SequenceDataset(
    env=args.dataset,
    horizon=1,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    use_padding=args.use_padding,
    seed=args.seed,
)

renderer = utils.MuJoCoRenderer(
    env=args.dataset
)

dc = DiffuserCritic(
    dataset=dataset,
    renderer=renderer,
    cond_dim=dataset.observation_dim,
    device=args.device,
    conditional=args.conditional,
    condition_dropout=args.condition_dropout,
    calc_energy=args.calc_energy,
    n_timesteps=args.n_diffusion_steps,
    clip_denoised=args.clip_denoised,
    condition_guidance_w=args.condition_guidance_w,
    beta_schedule=args.beta_schedule,
    warmup_steps=args.warmup_steps,
    maxq=args.maxq,
    alpha=args.alpha, 
    ema_decay=args.ema_decay,
    train_batch_size=args.batch_size,
    gradient_accumulate_every=args.gradient_accumulate_every,
    lr=args.lr,
    log_freq=args.log_freq,
    save_freq=int(args.n_train_steps // args.n_saves),
    label_freq=int(args.n_train_steps // args.n_saves),
    wandb=args.wandb,
)

if args.wandb:
    print('Wandb init...')
    wandb_dir = '/tmp/sykim/wandb'
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project=args.prefix.replace('/', '-'),
               entity='sungyoon',
               config=args,
               dir=wandb_dir,
               )
    wandb.run.name = f"{args.dataset}"
    
utils.report_parameters(dc.diffuser)
utils.report_parameters(dc.critic)
# utils.setup_dist()

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss = dc.diffuser.loss(*batch)
loss.backward()
print('✓')

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    dc.train(n_train_steps=args.n_steps_per_epoch)
    
if args.wandb:
    wandb.finish()