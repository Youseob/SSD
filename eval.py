import json
import wandb
import torch
import os
import numpy as np
import copy
from d4rl import reverse_normalized_score, get_normalized_score

import datasets
from dc.dc import DiffuserCritic
from dc.policy import *
import utils
from utils.arrays import to_torch, to_np

##############################################################################
################################ Config setup ################################
##############################################################################

class IterParser(utils.HparamEnv):
    dataset: str = 'FetchPickAndPlace-v1'
    config: str = 'config.fetch'
    experiment: str = 'evaluate'

iterparser = IterParser()

class Parser(utils.Parser):
    pid: int = 0
    cid: float = 0

args = Parser().parse_args(iterparser)

##############################################################################
################################### Setup ####################################
##############################################################################

env = datasets.load_environment(args.dataset)
env.seed(args.epi_seed)
action_dim = env.action_space.shape[0]
if 'Fetch' in args.dataset:
    observation_dim = env.observation_space['observation'].shape[0]
else:
    observation_dim = env.observation_space.shape[0]
horizon = args.horizon

dataset = datasets.SequenceDataset(
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    max_n_episodes=args.max_n_episodes,
    use_padding=args.use_padding,
    seed=args.seed,
)

if 'maze2d' in args.dataset:
    goal_dim = 2
    renderer = utils.Maze2dRenderer(env=args.dataset)
elif 'Fetch' in args.dataset:
    goal_dim = 3
    renderer = utils.FetchRenderer(env=args.dataset)
else:
    goal_dim = 1
    renderer = utils.MuJoCoRenderer(env=args.dataset)


dc = DiffuserCritic(
    dataset=dataset,
    renderer=renderer,
    goal_dim=goal_dim,
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

try: 
    has_object = dataset.env.has_object
except:
    has_object = False
    
if args.control == 'torque':
    policy = GoalTorqueControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
elif args.control == 'position':
    policy = GoalPositionControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
elif args.control == 'every':
    policy = SampleEveryControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
elif args.control == 'fetch':
    policy = FetchControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
else: 
    NotImplementedError(args.control)

## Init wandb
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

##############################################################################
############################## Start iteration ###############################
##############################################################################

state = env.reset()

## Set target and condition
if 'maze2d' in args.dataset:
    if args.multi: 
        print('Resetting target')
        env.set_target()
    ## set conditioning xy position to be the goal
    target = env._target
elif 'Fetch' in args.dataset:
    ## set conditioning xyz position to be the goal
    target = env.goal
else:
    ## set conditioning rtg to be the goal
    target = reverse_normalized_score(args.dataset, args.target_rtg)
    # target = dataset.normalizer(target, 'rtgs')
condition = torch.ones((1, horizon, 1)).to(args.device)
# condition[0, -1] = 1
gamma = dc.critic.gamma

total_reward = 0
rollout = []
rollout_sim = []
at_goal = False
for t in range(env.max_episode_steps):
    # samples = dc.diffuser(to_torch(state).unsqueeze(0), condition[t].reshape(1,1,1).repeat(1,args.horizon,1), to_torch(target).reshape(1,1))
    if 'maze2d' in args.dataset:
        at_goal = np.linalg.norm(state[:goal_dim] - target) <= 0.5
    elif 'Fetch' in args.dataset:
        at_goal = np.linalg.norm(state['achieved_goal'] - state['desired_goal']) <= 0.05
        state = state['observation']

    if args.increasing_condition:
        condition = torch.ones((1, horizon, 1)).to(args.device) * gamma ** (1 - ((t + horizon) / env.max_episode_steps))

    action = policy.act(state, condition, target, at_goal)

    # Store rollout for rendering
    if 'Fetch' in args.dataset:
        rollout_sim.append(copy.deepcopy(env.sim.get_state()))
    else:
        rollout.append(state[None, ].copy())
    
    # Step
    next_state, reward, done, _ = env.step(action)
    
    # If mujoco, decrease target rtg
    if 'Fetch' in args.dataset:
        reward += 1
        total_reward += reward
        score = total_reward
    elif 'maze2d' in args.dataset:
        total_reward += reward
        score = env.get_normalized_score(total_reward)
    else:
        if args.decreasing_target:
            # target = dataset.normalizer.unnormalize(target, 'rtgs')
            target -= reward
            # target = dataset.normalizer(target, 'rtgs')
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )
    
    if args.wandb:
        wandb.log({
            "reward": reward,
            "total_reward": total_reward,
            "score": score,
        }, step = t)

    if 'maze2d' in args.dataset:
        xy = next_state[:2]
        goal = env.unwrapped._target
        print(f'maze | pos: {xy} | goal: {goal}')
    elif 'Fetch' in args.dataset:
        ag = next_state['achieved_goal'][:3]
        goal = next_state['desired_goal']
        print(f'Fetch | ag: {ag} | goal: {goal}')
    
    if done:
        break
    state = next_state

# Rendering
if 'Fetch' in args.dataset:
    success = (reward == 1)
    print('success:', success)
    renderer.composite(f'{args.logbase}/{args.dataset}/{args.exp_name}/rollout.png', rollout_sim)
else:
    renderer.composite(f'{args.logbase}/{args.dataset}/{args.exp_name}/rollout.png', rollout)
    
# renderer.render_rollout(f'{args.logbase}/{args.dataset}/{args.exp_name}/rollout.mp4', rollout_sim)

if args.wandb:
    wandb.finish()