import json
import wandb
import torch
import os
import numpy as np
import copy
from utils.eval_module import main_kitchen
from dc.policy import *
import datasets
from dc.dc import DiffuserCritic
import utils


class IterParser(utils.HparamEnv):
    dataset: str = 'kitchen-mixed-v0'
    config: str = 'config.kitchen'
    experiment: str = 'evaluate'

iterparser = IterParser()

class Parser(utils.Parser):
    pid: int = 0
    cid: float = 0

args = Parser().parse_args(iterparser)
env = datasets.load_environment(args.dataset)
dataset = datasets.SequenceDataset(
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    max_n_episodes=args.max_n_episodes,
    use_padding=args.use_padding,
    termination_penalty=None,
    seed=args.seed,
)

horizon = args.horizon
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
goal_dim = 30
try: 
    has_object = dataset.env.has_object
except:
    has_object = False
    

dc = DiffuserCritic(
    dataset=dataset,
    renderer=None,
    goal_dim=30,
    device=args.device,
    dim_mults=args.dim_mults,
    conditional=args.conditional,
    condition_dropout=args.condition_dropout,
    calc_energy=args.calc_energy,
    n_timesteps=args.n_diffusion_steps,
    clip_denoised=args.clip_denoised,
    condition_guidance_w=args.condition_guidance_w,
    beta_schedule=args.beta_schedule,
    action_weight=args.action_weight,
    # warmup_steps=args.warmup_steps,
    maxq=args.maxq,
    alpha=args.alpha, 
    ema_decay=args.ema_decay,
    train_batch_size=args.batch_size,
    gradient_accumulate_every=args.gradient_accumulate_every,
    lr=args.lr,
    logdir=f'{args.logbase}/{args.dataset}/{args.exp_name}',
    diffusion_loadpath=f'{args.logbase}/{args.dataset}/{args.diffusion_loadpath}',
    log_freq=args.log_freq,
    save_freq=int(args.n_train_steps // args.n_saves),
    label_freq=int(args.n_train_steps // args.n_saves),
    wandb=args.wandb,
)
dc.load(args.diffusion_epoch)


policy = KitchenControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)

## Init wandb
if args.wandb:
    print('Wandb init...')
    wandb_dir = '/tmp/sykim/wandb'
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project=args.prefix.replace('/', '-'),
               entity='aaai2024',
               config=args,
               dir=wandb_dir,
               )
    wandb.run.name = f"{args.target_v}-{args.dataset}"

##############################################################################
############################## Start iteration ###############################
##############################################################################
state = env.reset()

succ_rate, undisc_returns, disc_returns, _ = main_kitchen(env, 10, policy, args.target_v)
output = {
    'success_rate': np.array(succ_rate).mean(),
    'returns': np.array(undisc_returns).mean(),
    'scores': np.array(disc_returns).mean(),
}

wandb.log(output)
wandb.finish()