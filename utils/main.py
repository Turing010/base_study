import argparse
import json
from datetime import datetime
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.env_temp import MECenv
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
import matplotlib.pyplot as plt

USE_CUDA = False  # torch.cuda.is_available()
# critic 输入obs要改 buffer

def pad_actions(actions):
    """
        Pad the actions with 10s to ensure all arrays have the same length.

        Args:
        - obs (list): List of lists of NumPy arrays representing actions.

        Returns:
        - np.ndarray: NumPy array with padded actions.
        """
    max_length = max(len(arr) for sublist in actions for arr in sublist)
    actions = [np.pad(a, ((0, 0), (0, max_length - len(a[0]))), mode='constant', constant_values=10) for a in actions]
    return actions
def make_parallel_env( n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = MECenv()#
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.n_rollout_threads, config.seed,
                            config.discrete_action)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,

                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0

    # 创建一个列表用于存储每个 episode 的奖励数据
    episode_rewards = []
    # 创建一个列表用于存储每个 episode 的奖励数据及对应episode索引用于写入meanreward文件
    data_episode_rewards = []
    # 在开始训练之前定义一个空的列表，用于存储每个时间步的动作数据
    action_data = []

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)###
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            # 向动作空间扩充至相同维度，填充10
            agent_actions=pad_actions(agent_actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            actions_list = [[agent_actions.tolist() for agent_actions in timestep_actions] for timestep_actions in
                            agent_actions]
            obs_list = [[obs.tolist() for obs in timestep_actions] for timestep_actions in
                        obs]
            next_obs_list = [[next_obs.tolist() for next_obs in timestep_actions] for timestep_actions in
                             next_obs]
            action_data.append({
                "episode": ep_i,
                "timestamp": et_i,
                "obs": obs_list,
                "rewards": rewards[0].tolist(),
                "next_obs": next_obs_list,
                "actions": actions_list
            })

            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)

        episode_rewards.append(np.mean(ep_rews))

        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        current_time = datetime.now().strftime("%Y-%m-%d")
        with open(f"maddpg_{current_time}.json", 'w') as f:
            for i, data_entry in enumerate(action_data):
                json.dump(data_entry, f)
                if i < len(action_data) - 1:
                    f.write(',\n')  # 添加逗号和换行符
                else:
                    f.write('\n')  # 最后一个元素不添加逗号，只添加换行符

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    # 在训练结束后将所有 episode 的奖励数据及episode索引保存到 meanrew.json 文件中
    with open('meanrew.json', 'w') as f:
        json.dump(data_episode_rewards, f)

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

    # 绘制奖励随 episode 变化的图像
    x_values = list(range(len(episode_rewards)))
    plt.plot(x_values, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward_Maddpg')
    plt.title('Reward_Maddpg')
    # 使用当前时间作为图片名字保存
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"Reward_Maddpg_{current_time}.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="examplemodel", type=str,
                        choices=['examplemodel', 'model1'],help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=1000, type=int)
    parser.add_argument("--init_noise_scale", default=1, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
