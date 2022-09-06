import os
import yaml
import random
import copy
import argparse

import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer

import a1sim
from a1sim.envs.env_builder import SENSOR_MODE

from gait.logger import WandbLogger, SAVE_REWARD_ITEMS, SAVE_STATE_ITEMS, SAVE_TERMIAL_ITEMS
from gait.model import GaitModel
from gait.wrapper import TianshouWrapper, TrajectoryGenerator

from path import *

SUBPATH_CONFIG = {  "reward":   "reward.yaml",
                    "ppo":      "ppo.yaml",
                    "experiment": "experiment.yaml",
                    "simulation": "simulation.yaml"}

LOGGER_TOPIC_LIST = SAVE_REWARD_ITEMS + SAVE_STATE_ITEMS + SAVE_TERMIAL_ITEMS

RANDOM_PARAM_DICT = {'random_dynamics':0, 'random_force':0}
DYNAMIC_PARAM_DICT = {'control_latency': 0.001, 'joint_friction': 0.025, 'spin_foot_friction': 0.2, 'foot_friction': 1}

## Train Gait Controller
def train(save_path, config_path):

    # Load training configuration.
    exp_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["experiment"]), 'r'), Loader=yaml.FullLoader)
    ppo_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["ppo"]), 'r'), Loader=yaml.FullLoader)
    rew_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["reward"]), 'r'), Loader=yaml.FullLoader)
    sim_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["simulation"]), 'r'), Loader=yaml.FullLoader)

    print("\r\nExperiment Info")
    print(exp_config)
    print("\r\nPPO CONFIG")
    print(ppo_config)
    print("\r\nReward CONFIG")
    print(rew_config)

    # Configure envirnoment setup.
    random_param = copy.copy(RANDOM_PARAM_DICT)
    dynamic_param = copy.copy(DYNAMIC_PARAM_DICT)
    sensor_mode = copy.copy(SENSOR_MODE)

    random_param['random_dynamics'] = sim_config["Random Dynamics"]
    random_param['random_force'] = sim_config["Random Force"]

    sensor_mode['dis'] = sim_config["Sensor Dis"]
    sensor_mode['motor'] = sim_config["Sensor Motor"]
    sensor_mode["imu"] = sim_config["Sensor IMU"]
    sensor_mode["contact"] = sim_config["Sensor Contact"]
    sensor_mode["footpose"] = sim_config["Sensor Footpose"]
    sensor_mode["dynamic_vec"] = sim_config["Sensor Dynamic"]
    sensor_mode["force_vec"] = sim_config["Sensor Exforce"]
    sensor_mode["noise"] = sim_config["Sensor Noise"]

    reward_param = copy.copy(rew_config)

    print("\r\nReward CONFIG")
    print(reward_param)
    print("\r\nRandom CONFIG")
    print(random_param)
    print("\r\nDynamics CONFIG")
    print(dynamic_param)
    print("\r\nSensor CONFIG")
    print(sensor_mode)

    # Create directories for saved files.
    policy_path = os.path.join(save_path, 'policy')
    config_path = os.path.join(save_path, 'config')
    video_path = os.path.join(save_path, 'video')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(policy_path, exist_ok=True)
    os.makedirs(config_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    # Save configuration files.
    with open(os.path.join(config_path, 'experiment.yaml'), 'w') as yaml_file:
        yaml.dump(exp_config, yaml_file, default_flow_style=False)
    with open(os.path.join(config_path, 'ppo.yaml'), 'w') as yaml_file:
        yaml.dump(ppo_config, yaml_file, default_flow_style=False)
    with open(os.path.join(config_path, 'reward.yaml'), 'w') as yaml_file:
        yaml.dump(rew_config, yaml_file, default_flow_style=False)
    with open(os.path.join(config_path, 'simulation.yaml'), 'w') as yaml_file:
        yaml.dump(sim_config, yaml_file, default_flow_style=False)

    cmd_trj = TrajectoryGenerator()

    # Checking an RL environment.
    env = a1sim.make_env(task="none",
                            random_param=random_param, reward_param=reward_param, dynamic_param=dynamic_param,  sensor_mode=sensor_mode, 
                            num_history=sim_config["State History"], render=False, terrain=None, cmd=cmd_trj,
                            normal=1)    
    obs_dim = env.observation_space.shape[0]
    sensor_dim = env.sensor_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0]

    # Action offset, scale
    act_scale = np.array([0.2,0.6,0.6]*4)
    act_offset = np.array([0., 0.0, 0.]*4) 

    # curr = CurriculumHandler(min_scale=0.1, max_scale=2.0, coeff=10e6)

    # Training environments
    train_envs = SubprocVectorEnv(
        [lambda: TianshouWrapper(a1sim.make_env(
                                    random_param=random_param, reward_param=reward_param, dynamic_param=dynamic_param, sensor_mode=sensor_mode, 
                                    num_history=sim_config["State History"], render=False, terrain=None, cmd=cmd_trj,
                                    normal=1), 
                                    action_scale=act_scale, action_offset=act_offset, max_step = exp_config["Steps Per Episode"])
            for _ in range(ppo_config["Training Envs"])],
        norm_obs=False)

    # Evaluation environments
    test_envs = SubprocVectorEnv(
        [lambda: TianshouWrapper(a1sim.make_env(
                                    random_param=random_param, reward_param=reward_param, dynamic_param=dynamic_param, sensor_mode=sensor_mode, 
                                    num_history=sim_config["State History"], render=False, terrain=None, cmd=cmd_trj,
                                    normal=1),
                                    action_scale=act_scale, action_offset=act_offset, max_step = exp_config["Steps Per Episode"]) 
            for _ in range(ppo_config["Test Envs"]-1)]
        + [lambda: TianshouWrapper(a1sim.make_env(
                                    random_param=random_param, reward_param=reward_param, dynamic_param=dynamic_param, sensor_mode=sensor_mode, 
                                    num_history=sim_config["State History"], render=False, terrain=None, cmd=cmd_trj,
                                    normal=1), 
                                    action_scale=act_scale, action_offset=act_offset, max_step = exp_config["Steps Per Episode"], video_path=video_path)],
        norm_obs=False, obs_rms=train_envs.obs_rms, update_obs_rms=False)

    # Seed
    random.seed(exp_config["Seed"])
    np.random.seed(exp_config["Seed"])
    torch.manual_seed(exp_config["Seed"])
    torch.cuda.manual_seed(exp_config["Seed"])    
    train_envs.seed()
    test_envs.seed()

    # Actor model
    net_a = GaitModel(sensor_dim, action_dim, lenHistory=sim_config["State History"], device=exp_config["Device"])
    actor = ActorProb(net_a, action_dim, max_action=action_max, unbounded=True, device=exp_config["Device"])
    actor.to(exp_config["Device"])

    # Critic model
    net_c = GaitModel(sensor_dim, action_dim, lenHistory=sim_config["State History"], device=exp_config["Device"])
    critic = Critic(net_c, device=exp_config["Device"])
    critic.to(exp_config["Device"])
    
    # Scale last policy layer to make initial actions have 0 mean and std.
    torch.nn.init.constant_(actor.sigma_param, ppo_config["Initial Sigma"])
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    # Optimizer
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=ppo_config["Learning Rate"])

    # Learning rate
    lr_scheduler = None
    if ppo_config["Learning Rate Decay"]:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            ppo_config["Step/Epoch"] / ppo_config["Step/Collect"]) * ppo_config["Epoch"]

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    # Action distribution for exploration
    def dist(*logits):
        try:
            return Independent(Normal(*logits), 1)
        except ValueError as e:
            raise ValueError from e

    # PPO policy
    policy = PPOPolicy( actor, critic, optim, dist, 
                        discount_factor=ppo_config["Gamma"],
                        gae_lambda=ppo_config["GAE Lambda"],
                        max_grad_norm=ppo_config["Max Grad Norm"],
                        vf_coef=ppo_config["Value Coefficient"], 
                        ent_coef=ppo_config["Entropy Coefficient"],
                        reward_normalization=ppo_config["Reward Nomalization"], 
                        action_scaling=True,
                        action_bound_method="clip",
                        lr_scheduler=lr_scheduler, 
                        action_space=env.action_space,
                        eps_clip=ppo_config["Epsilon Clip"],
                        value_clip=ppo_config["Value Clip"],
                        dual_clip=ppo_config["Dual Clip"], 
                        advantage_normalization=ppo_config["Advantage Normalization"],
                        recompute_advantage=ppo_config["Recompute Advantage"])

    # Data collector
    if ppo_config["Training Envs"] > 1:
        buffer = VectorReplayBuffer(ppo_config["Buffer Size"], len(train_envs))
    else:
        buffer = ReplayBuffer(ppo_config["Buffer Size"])
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # Log writer
    logger = WandbLogger(project = exp_config["Project"], task = exp_config["Task"], 
                        path=save_path, update_interval=1000,
                        reward_config=rew_config, ppo_config=ppo_config, experiment_config=exp_config,
                        actor=net_a, critic=net_c)

    # Policy recorder
    def save_fn(policy):
        torch.save(policy.state_dict(),
                   os.path.join(policy_path, 'policy.pth'))

    # Checkpoint recorder
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        torch.save({'model': policy.state_dict(),'optim': optim.state_dict(),},
                    os.path.join(policy_path, 'checkpoint_{}.pth'.format(epoch)))

    # Trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, ppo_config["Epoch"], ppo_config["Step/Epoch"],
        ppo_config["Repeat/Collect"], ppo_config["Test Envs"], ppo_config["Batch Size"],
        step_per_collect=ppo_config["Step/Collect"], save_fn=save_fn, save_checkpoint_fn=save_checkpoint_fn, logger=logger,
        test_in_train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait_policy",type=str,default="corl2022",
                        help="path for saving checkpoints, configuration and training logs. For example, --gait_policy=GAIT_POLICY will save checkpoints at ./save/rl_checkpoints/gait/GAIT_POLICY.")
    parser.add_argument("--config",type=str,default="gait",
                        help="path to a directory with the configuration files of RL hyper-parameters, reward, simulation setup, etc. For example, --config=CONFIG will save checkpoints at ./config/CONFIG.")
    args = parser.parse_args()
    save_path = os.path.join(PATH_CHECKPOINT_RL, "gait", args.gait_policy)
    train(save_path, os.path.join(PATH_CONFIG, args.config))