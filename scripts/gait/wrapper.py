
import datetime
import gym
import torch
import cv2
import numpy as np

from tianshou.data import Batch, to_numpy

RENDER_WIDTH = 480
RENDER_HEIGHT = 360

COMMAND_LIN_MAX = 1.5
COMMAND_LIN_MIN = 0
COMMAND_LIN_BASE = 1.0
COMMAND_LIN_DIS = 1.0*0.004 * (33./13.)

COMMAND_LAT_MAX = 0.5 * (33./13.)
COMMAND_LAT_MIN = -0.5 * (33./13.)
COMMAND_LAT_BASE = 0.0
COMMAND_LAT_DIS =1.0*0.004 * (33./13.)

COMMAND_YAW_MAX = 0.7 * (33./13.)
COMMAND_YAW_MIN = -0.7 * (33./13.)
COMMAND_YAW_BASE = 0.0
COMMAND_YAW_DIS = 1.0*0.008 * (33./13.)

COMMAND_SCALE = (33./13.)

## Trajectory handler for velocity commands at evaluating/deploying Gailt Controller
class TrajectoryHandler():
    def __init__(self):
        self._scale = COMMAND_SCALE
        self.reset()

    def reset(self):
        self.target_xy = np.zeros(2)
        self.target_yaw = 0.0
        return self.target_xy, self.target_yaw

    def get_target(self):
        return self.target_xy, self.target_yaw

    def set_scale(self, scale):
        self._scale = scale

    def set_target(self, action):
        self.target_xy[0] = action[0]/self._scale
        self.target_yaw = action[1]/self._scale

    def evaluate(self, robot):
        xyz_pos = robot.GetBasePosition()
        xyz_vel = robot.GetTrueLocalBaseVelocity()
        rpy_pos = robot.GetTrueBaseRollPitchYaw()
        rpy_vel = robot.GetTrueBaseRollPitchYawRate()
        errors = np.concatenate(((self._scale*self.target_xy - xyz_vel[0:2]), self._scale*self.target_yaw-rpy_vel[2]), axis=None)
        return {'errors':errors, 'linear':xyz_vel, 'angular':rpy_vel, 'position': xyz_pos, 'orientation': rpy_pos}

    @property
    def scale(self):
        return self._scale

    @property
    def tracking(self):
        return True


## Trajectory generator for random velocity commands at training Gailt Controller
class TrajectoryGenerator():
    def __init__(self, currilculum_handler=None):
        self._scale = COMMAND_SCALE
        self._curriculum = currilculum_handler
        self.reset()

    def reset(self):
        self.target_xy = np.array([COMMAND_LIN_BASE, COMMAND_LAT_BASE])/self._scale
        self.target_yaw = COMMAND_YAW_BASE/self._scale

        return self.target_xy, self.target_yaw

    def get_target(self):

        if self._curriculum != None:
            scale = self._curriculum.get_scale()
            self._curriculum.add_count()
        else:
            scale = 1.0

        self.target_xy[0] += np.random.normal(0., scale*COMMAND_LIN_DIS)/self._scale
        self.target_yaw += np.random.normal(0., scale*COMMAND_YAW_DIS)/self._scale
        self.target_xy[0] = np.clip(self.target_xy[0], COMMAND_LIN_MIN/self._scale, COMMAND_LIN_MAX/self._scale)
        self.target_yaw = np.clip(self.target_yaw, COMMAND_YAW_MIN/self._scale, COMMAND_YAW_MAX/self._scale)
        return self.target_xy, self.target_yaw

    @property
    def scale(self):
        return self._scale

    @property
    def tracking(self):
        return True


## Curriculum for RL training, not used
class CurriculumHandler():
    def __init__(self, min_scale=0.1, max_scale=2.0, coeff=10e6):
        assert(max_scale>min_scale)
        self._min = min_scale
        self._max = max_scale
        self._coeff = coeff
        self._count = 0
        self.reset()

    def reset(self):
        self._count = 0

    def add_count(self):
        self._count += 1

    def get_scale(self):
        scale = self._max + (self._min - self._max) * np.exp(-1.0*self._count/self._coeff)
        return scale


## Agent class for evaluating and deploying the Gait Controller model
class PPOAgent():
    def __init__(self, policy) -> None:
        self.policy = policy
        self.data = Batch(obs={}, act={})

    def reset(self):
        pass

    def predict(self,obs):
        self.data.obs = [obs]
        with torch.no_grad():
            result = self.policy(self.data)
        action = self.policy.map_action(to_numpy(result.act))
        return action


## Env Wrapper for distributed PPO of Tianshou, and recording utils
class TianshouWrapper(gym.Wrapper):
    def __init__(self, env, action_scale, action_offset=0, max_step=600, video_path=None):
        gym.Wrapper.__init__(self, env)

        self._video_save_path = video_path
        self.recorder = None
        self.max_step = max_step

        if self._video_save_path != None:
            self._recorder_reset = False
            self._render_width = RENDER_WIDTH
            self._render_height = RENDER_HEIGHT
            self._video_format = cv2.VideoWriter_fourcc(*'mp4v')

        self.action_scale = action_scale
        self.action_offset = action_offset


    def reset(self,**kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_step = 0

        if self._video_save_path !=None:
            self._recorder_reset = False

        return obs
    

    def step(self, action):
        obs, rew, done, info = self.env.step(self.action_scale * action + self.action_offset)
        self.num_step += 1

        if self._video_save_path != None:
            self._record_save()
            
        done = done or (self.num_step >= self.max_step)
        return obs, rew, done, info


    def close(self):
        self.env.close()


    def _record_reset(self):

        if self._video_save_path !=None:

            if self.recorder != None:
                self.recorder.release()
                del self.recorder
                self.recorder = None
                
            self.recorder = cv2.VideoWriter("{}/{}.mp4".format(self._video_save_path, datetime.datetime.now().strftime("%m%d_%H%M%S")),
                                            self._video_format, 30, (self._render_width, self._render_height))


    def _record_save(self):

        if self._video_save_path !=None:

            if not self._recorder_reset:
                self._record_reset()
                self._recorder_reset = True

            img = self.env.render(mode='rgb_array')[:, :, [2, 1, 0]]
            self.recorder.write(img)

