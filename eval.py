import json

import datasets
from dc.dc import DiffuserCritic
import utils
from utils.arrays import to_torch, to_np

class IterParser(utils.HparamEnv):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'
    experiment: str = 'evaluate'

iterparser = IterParser()

class Parser(utils.Parser):
    pid: int = 0
    cid: float = 0

args = Parser().parse_args(iterparser)
env = datasets.load_environment(args.dataset)
action_dim = env.action_space.shape[0]

dataset = datasets.SequenceDataset(
    env=args.dataset,
    horizon=1,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    use_padding=args.use_padding,
    seed=args.seed,
)

if 'maze2d' in args.dataset:
    renderer = utils.Maze2dRenderer(env=args.dataset)
else:
    renderer = utils.MuJoCoRenderer(env=args.dataset)


dc = DiffuserCritic(
    dataset=dataset,
    renderer=renderer,
    cond_dim=2,
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
    logdir=f'{args.logbase}/{args.dataset}/{args.exp_name}',
    diffusion_loadpath=f'{args.logbase}/{args.dataset}/{args.diffusion_loadpath}',
    log_freq=args.log_freq,
    save_freq=int(args.n_train_steps // args.n_saves),
    label_freq=int(args.n_train_steps // args.n_saves),
    wandb=args.wandb,
)

dc.load(args.diffusion_epoch)

state = env.reset()


print('Resetting target')
env.set_target()

## set conditioning xy position to be the goal
target = env._target

total_reward = 0
for t in range(env.max_episode_steps):
    state = env.state_vector().copy()
    samples = dc.diffuser(to_torch(state).unsqueeze(0), to_torch(target).unsqueeze(0))
    action = to_np(samples)[0, :action_dim]
    
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_state[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )
    
    if done:
        break
    state = next_state
    