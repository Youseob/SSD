import socket
from utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('condition_dropout', 'dr'),
    # ('conditional', 'cond'),
]


eval_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('batch_size', 'b'),
    ##
    ('maxq', 'maxq'),
    ('epi_seed', 's'),
]

base = {

    'diffusion': {
        ## model
        'dim_mults': [(1, 4, 8)],
        'horizon': [16],
        'n_diffusion_steps': [50],
        'action_weight': [1],
        'loss_weights': [None],
        'loss_discount': [1],
        'predict_epsilon': [True],
        'calc_energy': [False],

        ## dataset
        'termination_penalty': [None],
        'normalizer': ['LimitsNormalizer'],
        'preprocess_fns': [['mujoco_set_goals']],
        'use_padding': [True],
        'max_path_length': [1000],
        'max_n_episodes': [5000],
        
        ## diffuser
        'conditional': [True],
        'condition_guidance_w': [1.2],
        'condition_dropout': [0.25],
        'beta_schedule': ['cosine'],
        'clip_denoised': [True],

        ## serialization
        'logbase': ['/ext2/sykim/DC/logs'],
        # 'logbase': ['logs'],
        'prefix': ['dc/'],
        'exp_name': [watch(diffusion_args_to_watch)],

        ## training
        'seed': [0],
        'maxq': [False],
        'alpha': [1],
        'n_steps_per_epoch': [10000],
        'loss_type': ['l2'],
        'n_train_steps': [1e5],
        'warmup_steps': [4e4],
        'batch_size': [32],
        'lr': [2e-4],
        'gradient_accumulate_every': [2],
        'ema_decay': [0.995],
        # 'save_freq': [5000],
        'sample_freq': [5000],
        'log_freq': [100],
        'n_saves': [10],
        'save_parallel': [False],
        'n_reference': [50],
        'n_samples': [10],
        'bucket': [None],
        'device': ['cuda'],
        'wandb': [True],
    },
    
    'evaluate': {
        # 'guide': 'sampling.ValueGuide',
        'target_rtg': [1.0, 1.2, 1.4],
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
        'prefix': ['eval/dc'],
        'exp_name': [watch(eval_args_to_watch)],
        'vis_freq': [10],
        'max_render': [8],

        ## diffusion model
        'horizon': [16], #None,
        'n_diffusion_steps': [50],
        'condition_guidance_w': [1.2],
        'maxq': [True],
        'conditional': [True],

        ## loading
        'diffusion_loadpath': ['f:dc/H{horizon}_T{n_diffusion_steps}_gw{condition_guidance_w}'],
        'diffusion_epoch': [9999],

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


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        # 'n_guide_steps': 1,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}