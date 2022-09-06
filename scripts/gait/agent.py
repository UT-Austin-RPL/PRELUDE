import yaml
import gym
import os
import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal

from tianshou.policy import PPOPolicy
from tianshou.utils.net.continuous import ActorProb, Critic

from gait.model import GaitModel
from gait.wrapper import PPOAgent


SUBPATH_CONFIG = {  "reward":   "reward.yaml",
                    "ppo":      "ppo.yaml",
                    "experiment": "experiment.yaml",
                    "simulation": "simulation.yaml"}


## function for generating Gailt Controller
def build_agent(load_path, device=None):
    
    # configuration
    policy_path = load_path + "/policy"
    config_path = load_path + "/config"

    exp_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["experiment"]), 'r'), Loader=yaml.FullLoader)
    ppo_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["ppo"]), 'r'), Loader=yaml.FullLoader)
    sim_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["simulation"]), 'r'), Loader=yaml.FullLoader)

    if device == None:
        device=exp_config["Device"]

    action_space = gym.spaces.Box(low=np.array([-0.2, -0.7, -0.7, -0.2, -0.7, -0.7, -0.2, -0.7, -0.7, -0.2, -0.7, -0.7]),
                                       high=np.array([0.2, 0.7, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7, 0.7]), shape=(12,), dtype=np.float32)
    sensor_dim = 37
    action_dim = action_space.shape[0]
    action_max = action_space.high[0]

    # Initialize model, algorithm, agent, replay_memory
    net_a = GaitModel(sensor_dim, action_dim, lenHistory=sim_config["State History"], device=device)
    net_c = GaitModel(sensor_dim, action_dim, lenHistory=sim_config["State History"], device=device)

    # generate actor/critic models
    actor = ActorProb(net_a, action_dim, max_action=action_max, unbounded=True, device=device).to(device)
    critic = Critic(net_c, device=device).to(device)

    # generate an optimizer
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=ppo_config["Learning Rate"])

    # generate a learning rate scheduler
    lr_scheduler = None
    if ppo_config["Learning Rate Decay"]:
        max_update_num = np.ceil(
            ppo_config["Step/Epoch"] / ppo_config["Step/Collect"]) * ppo_config["Epoch"]

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    # generate a normal distribution for RL exploration
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    # generate a object for Gait Controller
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
                        action_space=action_space,
                        eps_clip=ppo_config["Epsilon Clip"],
                        value_clip=ppo_config["Value Clip"],
                        dual_clip=ppo_config["Dual Clip"], 
                        advantage_normalization=ppo_config["Advantage Normalization"],
                        recompute_advantage=ppo_config["Recompute Advantage"])

    # load the last checkpoints
    list_itr = []
    for file in os.listdir(policy_path):
        if file.endswith("pth"):
            if 'policy' in file:
                continue
            itr = file.replace('checkpoint_', "")
            itr = int(itr.replace('.pth', ""))
            list_itr.append(itr)

    checkpoint_path = '{}/checkpoint_{}.pth'.format(policy_path, max(list_itr))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])

    print('Loaded a RL checkpoint: {}'.format(checkpoint_path))

    # set the evaluation mode
    policy.eval()

    return PPOAgent(policy)