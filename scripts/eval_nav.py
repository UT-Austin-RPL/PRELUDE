import os
from a1sim.envs.external.constants import PATH_CONFIG
from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.config import config_factory
import numpy as np
import cv2
import argparse
import pickle
import yaml
import torch

from gait.agent import build_agent
from nav.environments import build_robot_env

from path import *

SUBPATH_CONFIG = {  "ppo":      "ppo.yaml",
                    "experiment": "experiment.yaml",
                    "simulation": "simulation.yaml"}

EXP_CONFIG = {'easy': {'env_type': [2], 'epoches': 50},
              'medium': {'env_type': [0, 3], 'epoches': 25},
              'hard': {'env_type': [1, 4], 'epoches': 25}}


##  Evaluate Navigation Controller with Gait Controller
def evaluate(nav_policy, gait_policy, difficulty, save_path=None, config_path=None, render=True,  seed=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    # Setup paths
    nav_path = "{}/{}/models/model_best_training.pth".format(PATH_CHECKPOINT_BC, nav_policy)
    gait_path = os.path.join(PATH_CHECKPOINT_RL, 'gait', gait_policy)
    env_data_path = os.path.join(PATH_DATA,'env_profile')
    if config_path == None:
        config_path = PATH_CONFIG

    # Configuration
    sim_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["simulation"]), 'r'), Loader=yaml.FullLoader)

    # Load checkpoints
    eval_policy = policy_from_checkpoint(ckpt_path=nav_path)[0]
    gait_agent = build_agent(gait_path, device=device)

    # Build environments
    env = build_robot_env(sim_config, gait_agent, render=render, record=True)

    # Record buffer
    successes = []
    distances = []

    # Loop for difficulties
    for env_type in EXP_CONFIG[difficulty]['env_type']:

        if save_path != None:
            os.makedirs("{}/env_{}".format(save_path, env_type), exist_ok=True)
            video_format = cv2.VideoWriter_fourcc(*'mp4v')

        # Loop for scenarios
        for eval_epoch in range(EXP_CONFIG[difficulty]['epoches']):

            # Setup scenarios
            done = False
            with open('{}/env_{}/{}.pickle'.format(env_data_path, env_type, eval_epoch),"rb") as f:
                profile = pickle.load(f)

            obs = env.reset(env_profile=profile)

            dic_record_array = {}
            dic_record_array['bird_view'] = []

            # Evaluation loop
            while not done:

                # Update Navigation Controller
                obs_dict = {
                    "agentview_rgb": 255.*np.transpose(obs["rgbd"][..., :3], (2, 0, 1)),
                    "agentview_depth": np.transpose(obs["rgbd"][..., 3:], (2, 0, 1)),
                    "yaw": np.array([obs["yaw"]])
                }
                action = eval_policy(obs_dict)

                obs, rew, done, info = env.step(action)
                dic_record_array['bird_view']+=env.record()

            successes.append(env.episode_eval_stat()["Success"])
            distances.append(env.episode_eval_stat()["Distance"])
            print('episode: env_{}/{}'.format(env_type, eval_epoch))
            print('current distance:', distances[-1])
            print('average distance:', np.average(distances))
            print('average success rate:', np.average(successes))
            print('==========================="')

            # Real-time recording, not used
            if save_path != None:
                recorder = cv2.VideoWriter("{}/env_{}/trial_{}.mp4".format(save_path, env_type, eval_epoch), video_format, 30, (480, 360))

                for img in dic_record_array['bird_view']:
                    recorder.write(img)
                
                recorder.release()

    print("Evaluation: ")
    print("\taverage: ", np.mean(distances))
    print("\tdeviation: ", np.std(distances))
    print("\tsuccess rate: ", np.mean(successes))
    
    if save_path != None:
        with open('{}/{}.txt'.format(save_path, difficulty),"w") as handle:
            for item in distances:
                handle.write("{}\n".format(item))
            handle.write("\naverage: {}\n".format(np.average(distances)))
            handle.write("\ndeviation: {}\n".format(np.std(distances)))
            handle.write("\nsuccess: {}\n".format(np.average(successes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait_policy",type=str,default="corl2022",
                        help="path for loading checkpoints, configuration and training logs. For example, --gait_policy=GAIT_POLICY will load checkpoints at ./save/rl_checkpoints/gait/GAIT_POLICY.")
    parser.add_argument("--nav_policy", type=str, default="bcrnn",
                        help="path for loading checkpoints, configuration and training logs. For example, --nav_policy=NAV_POLICY will load checkpoints at ./save/bc_checkpoints/NAV_POLICY.")
    parser.add_argument("--config",type=str,default="deploy",
                        help="path to a directory with the configuration files of simulation setup, etc. For example, --config=CONFIG will save checkpoints at ./config/CONFIG.")
    parser.add_argument("--difficulty", type=str, default="hard",
                        help="difficulty of simulation environments. Use one of these: 'easy', 'medium' and 'hard'.")

    args = parser.parse_args()

    nav_policy = args.nav_policy
    gait_policy = args.gait_policy
    difficulty = args.difficulty

    evaluate(nav_policy, gait_policy, difficulty)
