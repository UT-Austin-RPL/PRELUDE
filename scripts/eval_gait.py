import os
import yaml
import argparse
import random
import copy
import numpy as np

import a1sim
from a1sim.envs.env_builder import SENSOR_MODE

from gait.wrapper import TianshouWrapper, TrajectoryHandler
from gait.agent import build_agent

from utils.pid import PIDHandler

import cv2
import torch

from path import *

NUM_EVAL = 20

SUBPATH_CONFIG = {  "experiment": "experiment.yaml",
                    "simulation": "simulation.yaml"}

RANDOM_PARAM_DICT = {'random_dynamics':0, 'random_force':0}
DYNAMIC_PARAM_DICT = {'control_latency': 0.001, 'joint_friction': 0.025, 'spin_foot_friction': 0.2, 'foot_friction': 1}


##  Create 'x-linear' trajectories
def generate_linear_trj():

    trajectory = []
    time_step = 0.1
    time_max = 30
    cnt = 0

    x_linear = 0.7
    yaw_rate = 0.0

    pos = np.zeros(2)
    yaw = 0.

    while cnt*time_step < time_max:
        pos += np.array([np.cos(yaw) * x_linear, -np.sin(yaw)* x_linear])  * time_step
        yaw += yaw_rate * time_step

        trajectory.append({'time': cnt*time_step, 'x_linear': x_linear, 'yaw_rate': yaw_rate, 'position': np.copy(pos), 'yaw': yaw})
        cnt += 1
    return trajectory


##  Create sine trajectories for 'x-sine' and 'sine'
def generate_sine_trj(mode):

    trajectory = []
    time_step = 0.1
    time_max = 30
    cnt = 0

    x_linear = 0.7
    yaw_rate = 0.0

    pos = np.zeros(2)
    yaw = 0.

    while cnt*time_step < time_max:

        if mode == 'linear':
            x_linear = 0.7 + 0.3  *np.cos(cnt*time_step * 2*np.pi/15)
        else:
            yaw_rate = 0.3  *np.cos(cnt*time_step * 2*np.pi/15)

        pos += np.array([np.cos(yaw) * x_linear, -np.sin(yaw)* x_linear])  * time_step
        yaw += yaw_rate * time_step

        trajectory.append({'time': cnt*time_step, 'x_linear': x_linear, 'yaw_rate': yaw_rate, 'position': np.copy(pos), 'yaw': yaw})
        cnt += 1
    return trajectory


##  Create step-like trajectories for 'x-step' and 'zig-zag'
def generate_step_trj(mode):

    trajectory = []
    time_step = 0.1
    time_max = 30
    cnt = 0

    x_linear = 0.7
    yaw_rate = 0.0

    pos = np.zeros(2)
    yaw = 0.

    while cnt*time_step < time_max:

        if cnt*time_step < 5.0:
            if mode == 'linear':
                x_linear = 0.7
            else:
                yaw = 0.0 * np.pi
        elif cnt*time_step < 15.0:
            if mode == 'linear':
                x_linear = 1.0
            else:
                yaw = 0.16 * np.pi
        elif cnt*time_step < 25.0:
            if mode == 'linear':
                x_linear = 0.5
            else:
                yaw = -0.16 * np.pi
        else:
            if mode == 'linear':
                x_linear = 0.7
            else:
                yaw = 0.0 * np.pi

        pos += np.array([np.cos(yaw) * x_linear, -np.sin(yaw)* x_linear])  * time_step

        # yaw_rate = 0.3  *np.cos(cnt*time_step * 2*np.pi/15)
        yaw += yaw_rate * time_step

        trajectory.append({'time': cnt*time_step, 'x_linear': x_linear, 'yaw_rate': yaw_rate, 'position': np.copy(pos), 'yaw': yaw})
        cnt += 1
    return trajectory


##  Evaluate Gait Controller with the setpoint trajectories with PD position controller
def evaluate(load_path, trajectory, gait_mode='rl', save_path=None, config_path=None, render=True,  seed=0, 
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    # Configuration
    if config_path == None:
        config_path = PATH_CONFIG

    sim_config = yaml.load(open("{}/{}".format(config_path, SUBPATH_CONFIG["simulation"]), 'r'), Loader=yaml.FullLoader)

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

    reward_param = {}

    # Create an environment
    cmd_trj = TrajectoryHandler()

    env = a1sim.make_env(task="none", 
                            random_param=random_param, reward_param=reward_param, dynamic_param=dynamic_param,  sensor_mode=sensor_mode, 
                            num_history=sim_config["State History"], render=render, terrain=None, cmd=cmd_trj,
                            normal=1)    

    act_scale = np.array([0.2,0.6,0.6]*4)
    act_offset = np.zeros(12)

    env =TianshouWrapper(env, 
                        action_scale=act_scale, action_offset=act_offset, 
                        max_step = 1000000, 
                        video_path=None)
    agent = build_agent(load_path, device=device)

    env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_GUI,0)
    env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_SHADOWS,0)

    if save_path != None:
        video_format = cv2.VideoWriter_fourcc(*'mp4v')
        recorder = cv2.VideoWriter("{}/{}_eval_{}.mp4".format(save_path, gait_mode, seed), video_format, 30, (480, 360))

    # Create a position PD controller
    pid_vel = PIDHandler(gainP = 3.0, gainD = 0.6, gainI = 0)
    pid_yaw = PIDHandler(gainP = 3.0, gainD = 0.6, gainI = 0)
    pid_vel.reset()
    pid_yaw.reset()


    # Initiation
    obs = env.reset(x_noise=0)
    done = False

    tracking_step = 0
    tracking_log = []
    err_log = []

    # Visualization setup
    idVizBallSmallTarget = env.env.pybullet_client.createVisualShape(env.env.pybullet_client.GEOM_SPHERE, radius = 0.01, rgbaColor=[1, 0, 0, 1])
    idVizBallLargeTarget = env.env.pybullet_client.createVisualShape(env.env.pybullet_client.GEOM_SPHERE, radius = 0.05, rgbaColor = [1, 0, 0 ,1])

    tar_viz = [np.array([0, 0, 0.1])]*5
    id_tar_viz = []
    id_cam_viz = env.env.pybullet_client.createMultiBody(0, -1, idVizBallLargeTarget, tar_viz[0], [0, 0, 0, 1])

    for pos in tar_viz:
        id_tar_viz.append(
            env.env.pybullet_client.createMultiBody(0, -1, idVizBallSmallTarget, pos, [0, 0, 0, 1])
        )

    rec_time = 1.0/15

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    env.seed(seed)

    time_init = env.env.robot.GetTimeSinceReset()

    # Evaluation loop
    while not done:
        # Update time
        time_cur = env.env.robot.GetTimeSinceReset() - time_init

        # Update the PD controller
        xy = env.env.robot.GetBasePosition()[0:2]
        yaw = env.env.robot.GetTrueBaseRollPitchYaw()[2]
        rot_mat = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        error = rot_mat @ (trajectory[tracking_step]['position'] - xy)
        pid_vel.update(error[0])
        pid_yaw.update(error[1])
        err_log.append(error)

        cmd_trj.set_target(np.array([pid_vel.output(),
                                        pid_yaw.output()]))

        tracking_step_future = min(tracking_step+1, len(trajectory)-1)
        cam_pos_xy = trajectory[tracking_step]['position'] + (trajectory[tracking_step_future]['position'] - trajectory[tracking_step]['position']) * (1-(trajectory[tracking_step]['time']-time_cur)/0.1)
        cam_pos = np.array([0, 0, 0.1])
        cam_pos[0:2] = cam_pos_xy
        env.env.pybullet_client.resetBasePositionAndOrientation(id_cam_viz , cam_pos, [0, 0, 0, 1])

        # Real-time recording, not used
        if save_path != None and rec_time < time_cur:

            view_matrix = env.env.pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition = cam_pos + np.array([-0.3, 0, 0]),
                distance = 1.0,
                roll = 0,
                pitch = -45,
                yaw = 0,
                upAxisIndex=2)
            proj_matrix = env.env.pybullet_client.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(480) / 360,
                nearVal=0.1,
                farVal=100)
            (_, _, rgb, depth, _) = env.env.pybullet_client.getCameraImage(
                width=480,
                height=360,
                renderer=env.env.pybullet_client.ER_TINY_RENDERER,
                viewMatrix=view_matrix,
                shadow=0,
                projectionMatrix=proj_matrix)

            img = rgb[:,:,[2, 1, 0]]

            recorder.write(img)
            rec_time += 1.0/30

        # Update footprints, logs
        if trajectory[tracking_step]['time'] < time_cur:
            tracking_log.append(cmd_trj.evaluate(env.env.robot))

            if tracking_step < len(trajectory)-1:
                tracking_step += 1
                tar_viz[1:] = tar_viz[0:-1]
                tar_viz[0] = np.array([trajectory[tracking_step]['position'][0], trajectory[tracking_step]['position'][1], 0.1])

                for idx in range(len(id_tar_viz)):
                    env.env.pybullet_client.resetBasePositionAndOrientation(id_tar_viz[idx] , tar_viz[idx], [0, 0, 0, 1])

            else:
                break

        # Step environments
        action = agent.predict(obs)
        obs, rew, done, info = env.step(action)

        if np.sum(np.array(error[0:2])**2)>1:
            done = True

    print('errors: ', np.sqrt(np.nanmean(np.sum(np.array(err_log)**2, axis=1))))
    print('complemeted: ', True if not done else False)
    print('==========================="')

    if save_path != None:
        recorder.release()

    return tracking_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait_policy",type=str,default="policy_20220125_finished",
                        help="path for loading checkpoints, configuration and training logs. For example, --gait_policy=GAIT_POLICY will save checkpoints at ./save/rl_checkpoints/gait/GAIT_POLICY.")
    parser.add_argument("--config",type=str,default="deploy",
                        help="path to a directory with the configuration files of simulation setup, etc. For example, --config=CONFIG will save checkpoints at ./config/CONFIG.")
    parser.add_argument("--trajectory", type=str, default="x-linear",
                        help="shape of trajectories for the unit testing. Use one of these: 'x-linear', 'x-sine', 'x-step', 'sine' and 'zig-zag'.")
    args = parser.parse_args()
    path_load = os.path.join(PATH_CHECKPOINT_RL, "gait", args.gait_policy)

    if args.trajectory == 'sine':
        trj = generate_sine_trj('yaw')
    elif args.trajectory == 'x-sine':
        trj = generate_sine_trj('linear')
    elif args.trajectory == 'zig-zag':
        trj = generate_step_trj('yaw')
    elif args.trajectory == 'x-step':
        trj = generate_step_trj('linear')
    else:
        trj = generate_linear_trj()

    for idx in range(NUM_EVAL):
        data_logs = evaluate(path_load, trajectory=trj, config_path=PATH_CONFIG, render=False, seed = idx)
