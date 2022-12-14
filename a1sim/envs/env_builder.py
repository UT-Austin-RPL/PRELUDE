# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for building environments."""
from a1sim.envs import locomotion_gym_env
from a1sim.envs import locomotion_gym_config
from a1sim.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from a1sim.envs.env_wrappers import trajectory_generator_wrapper_env
from a1sim.envs.env_wrappers import simple_openloop
from a1sim.envs.env_wrappers import simple_forward_task
from a1sim.envs.sensors import robot_sensors
from a1sim.robots import a1
from a1sim.robots import robot_config

SENSOR_MODE = {"dis":1,"motor":1,"imu":1,"contact":1,"footpose":0}

def build_regular_env(robot_class,
                      motor_control_mode,
                      sensor_mode,
                      normal=0,
                      enable_rendering=False,
                      on_rack=False,
                      filter = 0,
                      action_space = 0,
                      num_action_repeat=13,
                      action_limit=(0.75, 0.75, 0.75),
                      wrap_trajectory_generator=True):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = num_action_repeat
  sim_params.enable_action_interpolation = False
  if filter:
    sim_params.enable_action_filter = True
  else:
    sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  sim_params.robot_on_rack = on_rack
  dt = sim_params.num_action_repeat*sim_params.sim_time_step_s

  gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)

  sensors = []
  noise = True if ("noise" in sensor_mode and sensor_mode["noise"]) else False
  if sensor_mode["dis"]:
    sensors.append(robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True,normal=normal,noise=noise))
  if sensor_mode["imu"]==1:
    sensors.append(robot_sensors.IMUSensor(channels=["R", "P", "Y","dR", "dP", "dY"],normal=normal,noise=noise))
  elif sensor_mode["imu"]==2:
    sensors.append(robot_sensors.IMUSensor(channels=["dR", "dP", "dY"],noise=noise))
  if sensor_mode["motor"]==1:
    sensors.append(robot_sensors.MotorAngleAccSensor(num_motors=a1.NUM_MOTORS,normal=normal,noise=noise,dt=dt))
  elif sensor_mode["motor"]==2:
    sensors.append(robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS,noise=noise))
  if sensor_mode["contact"] == 1:
    sensors.append(robot_sensors.FootContactSensor())
  elif sensor_mode["contact"] == 2:
    sensors.append(robot_sensors.SimpleFootForceSensor())
  if sensor_mode["footpose"]:
    sensors.append(robot_sensors.FootPoseSensor(normal=normal))

  task = simple_forward_task.SimpleForwardTask(None)

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            robot_class=robot_class,
                                            robot_sensors=sensors,
                                            task=task)

  env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  
  if (motor_control_mode == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env,
        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
            action_limit=action_limit,action_space=action_space))

  return env