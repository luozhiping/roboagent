## "Copyright (c) Meta Platforms, Inc. and affiliates"


import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
from utils import load_data, load_test_data
from utils import compute_dict_mean, set_seed, detach_dict
from constants import CAMERA_NAMES, TEXT_EMBEDDINGS

from policy import ACTPolicy, CNNMLPPolicy

import click
import numpy as np
import time
import os


def main(args):
    np.set_printoptions(suppress = True)

    task_emb = TEXT_EMBEDDINGS[0]
    task_emb = np.asarray(task_emb)
    task_emb = torch.from_numpy(task_emb).float()
    if torch.cuda.is_available():
        task_emb = task_emb.cuda()
    task_emb = task_emb.unsqueeze(0)

    ## robohive args
    # env_name = args['env_name']

    mode = args['mode']
    horizon = args['horizon']
    num_repeat = args['num_repeat']
    render = args['render']
    camera_name = args['camera_name']
    frame_size = args['frame_size']
    output_dir = args['output_dir']
    output_name = args['output_name']
    save_paths = args['save_paths']
    compress_paths = args['compress_paths']
    plot_paths = args['plot_paths']
    env_args = args['env_args']
    noise_scale = args['noise_scale']

    # command line parameters
    # is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    # dataset_dir = args['dataset_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # fixed parameters
    num_episodes = 200  ## VHANGE IT
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1}
    else:
        raise NotImplementedError
    torch.set_printoptions(precision=4, sci_mode=False)
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args['lr'],
        'real_robot': 'TBD',
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg']
    }

    policy_config['camera_names'] = CAMERA_NAMES
    config['camera_names'] = CAMERA_NAMES
    config['real_robot'] = True
    config['episode_len'] = 100

    ckpt_names = [f'train1130/policy_best.ckpt']
    ckpt_name = ckpt_names[0]
    # eval_bc(config, ckpt_name, save_episode=True)

    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'main'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
    print(loading_status)
    if torch.cuda.is_available():
        policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'train1130/dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks


    image_list = []  # for visualization
    qpos_list = []
    target_qpos_list = []
    rewards = []

    train_dataloader, stats, is_sim = load_test_data("./data", 8,
                                                                     1, 1, True, stats)

    print(len(train_dataloader))
    all_test = []
    all_data = []
    for batch_idx, data in enumerate(train_dataloader):
        # if batch_idx != 2 and batch_idx != 1:
        #     continue
#        if batch_idx == 5:
#            break
        print('**************:', batch_idx)
        with torch.inference_mode():
            image_data, qpos_data, action_data, is_pad, task_emb = data
            if torch.cuda.is_available():
                image_data, qpos_data, action_data, is_pad, task_emb = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), task_emb.cuda()
            image_data = image_data[0]
            qpos_data  = qpos_data[0]
            action_data = action_data[0]

            for t in range(image_data.shape[0]):
                begin = time.time()
                all_actions = policy(qpos_data[t].unsqueeze(0), image_data[t].unsqueeze(0), task_emb=task_emb)
                print(t, (begin - time.time()))
                tt1 = post_process(all_actions[0][0].cpu())[:6]
                dd1 = post_process(action_data[t].cpu())[:6]
                tt2 = post_process(all_actions[0][0].cpu())[6] * 450 + 900
                dd2 = post_process(action_data[t].cpu())[6] * 450 + 900
                print('test', tt1, tt2)
                print('data', dd1, dd2)
                all_test.append(tt1.numpy())
                all_data.append(dd1.numpy())
                # print(torch.abs(tt1 - dd1), torch.mean(torch.abs(tt1 - dd1)))
                # if t == 2:
                #     break
            # exit(0)
    all_test = np.array(all_test)
    all_data = np.array(all_data)
    offset = all_data - all_test
    offset_mean = np.mean(offset, axis=0)
    correct = (offset < 0.5).sum(axis=0) / offset.shape[0]
    print('^^^^^^^^^^^^^^^^^^^^^^')
    print('offset_mean: ', offset_mean)
    print('correct_rate: ', correct)
        # break
            ### query policy
            # if config['policy_class'] == "ACT":
            #
            # if t % query_frequency == 0:
            #     print('SAMPLED ACTION')
            #     temporal_agg = True
            #     if temporal_agg:
            #         all_time_actions[[t], t:t + num_queries] = all_actions
            #         actions_for_curr_step = all_time_actions[:, t]
            #         actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            #         actions_for_curr_step = actions_for_curr_step[actions_populated]
            #         k = 0.01
            #         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            #         exp_weights = exp_weights / exp_weights.sum()
            #         exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
            #         raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            #         print('TEMPORAL AGG')
            #     ### post-process actions
            #     raw_action = raw_action.squeeze(0).numpy()
            #     action = post_process(raw_action)
            #     target_qpos = action
            #
            #     act = target_qpos
            #     # add gaussian noise
            #     # act = act + env.env.np_random.normal(loc=0.0, scale=0.025, size=len(act))
            #     print(act)
            #     print(f"STEP: {t}")


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
    if torch.cuda.is_available():
        curr_image = curr_image.cuda()
    return curr_image


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    if torch.cuda.is_available():
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=False, default="/mnt/raid5/data/roboset/v0.4/setting_table_close_drawer_scene_1/")
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False,
                        default="ACT")

    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=2)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False, default=1000)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=False, default=1e-04)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False, default=10)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False, default=256)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False,
                        default=2048)
    parser.add_argument('--temporal_agg', action='store', type=bool, default=True)
    # parser.add_argument('-e', '--env_name', type=str, help='environment to load', required=True,
    #                     default='rpFrankaRobotiqData-v0')

    parser.add_argument('-m', '--mode', type=click.Choice(['record', 'render', 'playback', 'recover', 'policy']),
                        help='How to examine rollout', default='policy')
    parser.add_argument('-hor', '--horizon', type=int, help='Rollout horizon, when mode is record', default=100)
    parser.add_argument('-num_repeat', '--num_repeat', type=int, help='number of repeats for the rollouts', default=10)
    parser.add_argument('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']),
                        help='visualize onscreen or offscreen', default='none')
    parser.add_argument('-c', '--camera_name', type=str, default=[None, ], help=('list of camera names for rendering'))
    parser.add_argument('-fs', '--frame_size', type=tuple, default=(424, 240), help=('Camera frame size for rendering'))
    parser.add_argument('-o', '--output_dir', type=str, default='/checkpoint/homanga/cactiv2/robohivelogs',
                        help=('Directory to save the outputs'))
    parser.add_argument('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
    parser.add_argument('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
    parser.add_argument('-cp', '--compress_paths', type=bool, default=True,
                        help=('compress paths. Remove obs and env_info/state keys'))
    parser.add_argument('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
    parser.add_argument('-ea', '--env_args', type=str, default="{\'is_hardware\':True}",
                        help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
    parser.add_argument('-ns', '--noise_scale', type=float, default=0.0, help=('Noise amplitude in randians}"'))

    parser.add_argument('--task_name', type=str, default='open_drawer', help=('task name for multitask'))
    # add this for multi-task embedding condition
    parser.add_argument('--multi_task', action='store_true')

    main(vars(parser.parse_args()))

# python evaluate.py -e rpFrankaRobotiqData-v0 -p /checkpoint/jayvakil/v0.4/setting_table_close_drawer_scene_1/setting_table_close_drawer_scene_1_20230308-120120.h5 -m playback -f RoboSet -r none

# python eval_robot.py -e rpFrankaRobotiqDataRP04-v0 --ckpt_dir ckpt/ --chunk_size $CHUNK -ns 0.01 --num_repeat 10

# python eval_robot.py -e rpFrankaRobotiqDataRP02-v0 --ckpt_dir ckpt/rp02_manga_policies/chunk20/drawer_close --chunk_size 20 --num_repeat 10 --task_name close_drawer

# python eval_robot_multi_task.py -e rpFrankaRobotiqDataRP02-v0 --ckpt_dir ckpt/april7multitask --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --num_repeat 10 --task_name pick_butter --multi_task
