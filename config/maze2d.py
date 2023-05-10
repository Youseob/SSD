import socket
from utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('maxq', 'maxq'),
    ('conditional', 'cond'),
]


eval_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
    ('epi_seed', 's'),
]

base = {

    'diffusion': {
        ## model
        'horizon': [256],
        'n_diffusion_steps': [256],
        'action_weight': [1],
        'loss_weights': [None],
        'loss_discount': [1],
        'predict_epsilon': [True],
        'calc_energy': [False],

        ## dataset
        'termination_penalty': [None],
        'normalizer': ['LimitsNormalizer'],
        'preprocess_fns': [['her_maze2d_set_terminals']],
        # 'preprocess_fns': [['maze2d_set_terminals']],
        'use_padding': [False],
        'max_path_length': [400],
        'max_n_episodes': [100000],
        
        ## diffuser
        'conditional': [False],
        'condition_guidance_w': [1.2],
        'condition_dropout': [0.25],
        'beta_schedule': ['cosine'],
        'clip_denoised': [True],

        ## serialization
        # 'logbase': ['/ext2/sykim/DC/logs'],
        'logbase': ['logs'],
        'prefix': ['dc/'],
        'exp_name': [watch(diffusion_args_to_watch)],

        ## training
        'seed': [0],
        'maxq': [False],
        'alpha': [1],
        'n_steps_per_epoch': [10000],
        'loss_type': ['l2'],
        'n_train_steps': [1e5],
        'warmup_steps': [4e5],
        'batch_size': [32],
        'lr': [2e-4],
        'gradient_accumulate_every': [2],
        'ema_decay': [0.995],
        # 'save_freq': [5000],
        'sample_freq': [5000],
        'log_freq': [100],
        'n_saves': [5],
        'save_parallel': [False],
        'n_reference': [50],
        'n_samples': [10],
        'bucket': [None],
        'device': ['cuda'],
        'wandb': [True],
    },
    
    'evaluate': {
        # 'guide': 'sampling.ValueGuide',
        'target_rtg': [0.0, 0.8, 1.0, 1.2, 1.4],
        'decreasing_target_rtg': [True],
        # 'policy': ['sampling.DDPolicyV2'],
        # 'max_episode_length': [1000],
        'batch_size': [1],
        # 'preprocess_fns': [['maze2d_set_terminals']],
        'device': ['cuda'],
        'epi_seed': [0, 1, 2, 3, 4], 
        'wandb': [True],

        ## sample_kwargs
        'n_guide_steps': [2],
        'scale': [0.1],
        't_stopgrad': [2],
        'scale_grad_by_std': [True],
        'n_initial_steps': [1],
        'update_policy_every': [2],

        ## serialization
        'loadbase': [None],
        # 'logbase': ['./logs'],
        'logbase': ['/ext2/sykim/DC/logs'],
        'prefix': ['eval/release'],
        'exp_name': [watch(eval_args_to_watch)],
        'vis_freq': [10],
        'max_render': [8],

        ## diffusion model
        'horizon': [128], #None,
        'n_diffusion_steps': [64],
        'maxq': [False],
        'conditional': [False],

        ## loading
        'diffusion_loadpath': ['f:dc/H{horizon}_T{n_diffusion_steps}_maxq{maxq}_cond{conditional}'],
        'diffusion_epoch': [19999],

        'verbose': [False],
        'suffix': ['0'],
    },
    
    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(eval_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': [128],
        'n_diffusion_steps': [64],
    },
    'evaluate': {
        'horizon': [128],
        'n_diffusion_steps': [64],
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': [384],
        'n_diffusion_steps': [256],
    },
    'evaluate': {
        'horizon': [384],
        'n_diffusion_steps': [256],
    },
}