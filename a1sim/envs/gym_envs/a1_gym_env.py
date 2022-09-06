# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
import gym
from a1sim.envs import env_builder
from a1sim.robots import a1
from a1sim.robots import robot_config
from a1sim.envs.prelude_gym_env import Prelude_Wrapper, REWARD_PARAM_DICT, RANDOM_PARAM_DICT, DYNAMIC_PARAM_DICT

import numpy as np

class A1Hybrid(a1.A1):

  def __init__(self, 
                pybullet_client,
                # urdf_filename=None,
                enable_clip_motor_commands=False,
                time_step=0.001,
                action_repeat=10,
                sensors=None,
                control_latency=0.002,
                on_rack=False,
                enable_action_interpolation=True,
                enable_action_filter=False,
                motor_control_mode=None,
                reset_time=1,
                allow_knee_contact=False,
                ):

    self._pos_commands = np.zeros(12)
    self._tor_commands = np.zeros(12)
    self._hybrid_modes = np.ones(12)

    super().__init__(
                    pybullet_client,
                    enable_clip_motor_commands=enable_clip_motor_commands,
                    time_step=time_step,
                    action_repeat=action_repeat,
                    sensors=sensors,
                    control_latency=control_latency,
                    on_rack=on_rack,
                    enable_action_interpolation=True,
                    enable_action_filter=False,
                    motor_control_mode=motor_control_mode,
                    reset_time=reset_time,
                    allow_knee_contact=allow_knee_contact,
                    )


  def SetDirectCommands(self, pos_commands, tor_commands, hybrid_modes):
    self._pos_commands = pos_commands
    self._tor_commands = tor_commands
    self._hybrid_modes = hybrid_modes

  def ApplyAction(self, motor_commands, control_mode):
    """Apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    self.last_action_time = self._state_action_counter * self.time_step

    q, qdot = self._GetPDObservation()
    qdot_true = self.GetTrueMotorVelocities()

    motor_init = np.array([0, 0.9, -1.8] * 4)
    
    pos_commands = self._hybrid_modes * self._motor_direction * (self._pos_commands + self._motor_offset + motor_init) + (1-self._hybrid_modes) * q
    tor_commands = (1-self._hybrid_modes) * self._motor_direction * self._tor_commands

    pos_actual_torque, pos_observed_torque = self._motor_model.convert_to_torque(
        pos_commands, q, qdot, qdot_true, control_mode)

    tor_actual_torque = self._motor_model._strength_ratios * tor_commands
    tor_observed_torque = self._motor_model._strength_ratios * tor_commands

    actual_torque = self._hybrid_modes * pos_actual_torque + (1-self._hybrid_modes) * tor_actual_torque
    observed_torque = self._hybrid_modes * pos_observed_torque + (1-self._hybrid_modes) * tor_observed_torque

    # May turn off the motor
    self._ApplyOverheatProtection(actual_torque)

    # The torque is already in the observation space because we use
    # GetMotorAngles and GetMotorVelocities.
    self._observed_motor_torques = observed_torque

    # Transform into the motor space when applying the torque.
    self._applied_motor_torque = np.multiply(actual_torque,
                                             self._motor_direction)
    motor_ids = []
    motor_torques = []

    for motor_id, motor_torque, motor_enabled in zip(
        self._motor_id_list, self._applied_motor_torque,
        self._motor_enabled_list):
      if motor_enabled:
        motor_ids.append(motor_id)
        motor_torques.append(motor_torque)
      else:
        motor_ids.append(motor_id)
        motor_torques.append(0)
    self._SetMotorTorqueByIds(motor_ids, motor_torques)

    return motor_torques


class A1GymEnv(gym.Env):
  """A1 environment that supports the gym interface."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self,
               action_limit=(0.75, 0.75, 0.75),
               render=False,
               on_rack=False,
               sensor_mode = env_builder.SENSOR_MODE,
               normal = 0,
               filter_ = 0,
               action_space =0,
               reward_param = REWARD_PARAM_DICT,
               random_param = RANDOM_PARAM_DICT,
               dynamic_param = DYNAMIC_PARAM_DICT,
               num_history=48,
               terrain=None,
               cmd=None,
               control_mode='policy',
               **kwargs):

    if control_mode == 'mpc':
      num_action_repeat=5
      robot_class=A1Hybrid
    else:
      num_action_repeat=13
      robot_class=a1.A1

    self._env = env_builder.build_regular_env(
        robot_class,
        motor_control_mode = robot_config.MotorControlMode.POSITION,
        normal=normal,
        num_action_repeat=num_action_repeat,
        enable_rendering=render,
        action_limit=action_limit,
        sensor_mode = sensor_mode,
        filter = filter_,
        action_space = action_space,
        on_rack=on_rack)

    self.render = self._env.render

    self._env = Prelude_Wrapper(
        env = self._env,
        reward_param = reward_param,
        sensor_mode = sensor_mode,
        random_param = random_param,
        dynamic_param = dynamic_param,
        num_action_repeat=num_action_repeat,
        num_history=num_history,
        terrain=terrain,
        cmd=cmd,
    )
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space
    self.sensor_mode =sensor_mode

  def step(self, action,**kwargs):
    return self._env.step(action,**kwargs)

  def reset(self,**kwargs):
    return self._env.reset(**kwargs)

  def close(self):
    self._env.close()

  def __getattr__(self, attr):
    return getattr(self._env, attr)