import gym
import numpy as np
import collections
import a1sim.envs.external.terrain as terrain_loader

import copy

REWARD_PARAM_DICT = {'Reward Tracking':0, 'Reward Balance':0, 'Reward Gait':0, 'Reward Energy':0, 'Reward Fail':0, 'Reward Badfoot':0, 'Reward Footcontact':0}
RANDOM_PARAM_DICT = {'random_dynamics':0, 'random_force':0}
DYNAMIC_PARAM_DICT = {'control_latency': 0.04, 'joint_friction': 0.025, 'spin_foot_friction': 0.2, 'foot_friction': 1}

def Prelude_Wrapper(env, sensor_mode, num_history=48, num_action_repeat=13, reward_param=None, random_param=None, dynamic_param = None, terrain=None, cmd=None):
    env = SimulationWrapper(env=env, terrain=terrain, cmd=cmd)
    env = DynamicsWrapper(env=env, random_param=random_param, dynamic_param=dynamic_param)
    env = ObservationWrapper(env=env, sensor_mode=sensor_mode)
    env = ObsHistoryWrapper(env=env, num_history=num_history)
    env = RewardWrapper(env=env, reward_param=reward_param, num_action_repeat=num_action_repeat)
    return env

HEIGHT_CONTACT = np.array([[0.08, 0, 0.32]]*4)

class SimulationWrapper(gym.Wrapper):
    def __init__(self, env, terrain, cmd):
        gym.Wrapper.__init__(self, env) 
        self.pybullet_client = self.env.pybullet_client
        self.load_cmd = cmd

        if self.load_cmd.tracking:
            self.observation_space = gym.spaces.Box(low=np.concatenate((self.env.observation_space.low, [-1, -1, -1]), axis=None),
                                                    high=np.concatenate((self.env.observation_space.high, [1, 1, 1]), axis=None),
                                                    shape=(self.env.observation_space.shape[0] + 3 ,), dtype=np.float32)
        else:
            self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.robot = self.env.robot

        self.load_terrain = terrain
        if self.load_terrain !=None:
            if self.load_terrain == 'test':
                self.typeTerrain = terrain_loader.TERRAIN_TEST
            elif self.load_terrain == 'obstacles':
                pass
            else:
                terrain_loader.loadTerrain(terrain)
                self.typeTerrain = terrain_loader.TERRAIN_TRAINING
            self.modelTerrain = None


    def reset(self,**kwargs):

        if 'hard_reset' in kwargs.keys():
            self.env.set_hard_reset(kwargs['hard_reset'])

        obs,info = self.env.reset()
        self.robot = self.env.robot
        
        dic_name_to_id = {}

        for id_joint in range(self.pybullet_client.getNumJoints(self.robot.quadruped)):
            jointInfo = self.pybullet_client.getJointInfo(self.robot.quadruped, id_joint)
            dic_name_to_id[jointInfo[1].decode('UTF-8')] = jointInfo[0]

        ## Overwrite Google's colors
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['FR_hip_fixed'], rgbaColor=(0.2, 0.3, 0.3, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['FL_hip_fixed'], rgbaColor=(0.2, 0.3, 0.3, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['RR_hip_fixed'], rgbaColor=(0.2, 0.3, 0.3, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['RL_hip_fixed'], rgbaColor=(0.2, 0.3, 0.3, 1.0))

        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['FR_upper_joint'], rgbaColor=(0.8, 0.4, 0.0, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['FL_upper_joint'], rgbaColor=(0.8, 0.4, 0.0, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['RR_upper_joint'], rgbaColor=(0.8, 0.4, 0.0, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['RL_upper_joint'], rgbaColor=(0.8, 0.4, 0.0, 1.0))

        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['FR_hip_joint'], rgbaColor=(1.0, 1.0, 1.0, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['FL_hip_joint'], rgbaColor=(1.0, 1.0, 1.0, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['RR_hip_joint'], rgbaColor=(1.0, 1.0, 1.0, 1.0))
        self.pybullet_client.changeVisualShape(self.robot.quadruped, dic_name_to_id['RL_hip_joint'], rgbaColor=(1.0, 1.0, 1.0, 1.0))

        if self.load_terrain !=None:
            # self.pybullet_client.removeBody(self.env.get_ground())
            del self.env._world_dict['ground']
            del self.modelTerrain
            self.modelTerrain = terrain_loader.modelTerrain(self.pybullet_client, typeTerrain = self.typeTerrain)
            self.env._world_dict['ground'] = self.modelTerrain.idTerrain

        info['fdb_xy'] = self.robot.GetTrueLocalBaseVelocity()[0:2]

        if self.load_cmd.tracking:
            cmd_xy, cmd_yaw = self.load_cmd.reset()
            info['cmd_xy'] = cmd_xy
            info['cmd_yaw'] = cmd_yaw
            info['cmd_scale'] = self.load_cmd.scale
            obs = np.concatenate((cmd_xy, cmd_yaw, obs), axis=None)
        else:
            obs = np.concatenate((self.load_cmd.get_target(), obs), axis=None)

        return obs, info
    
    
    def step(self,action,**kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)

        info['fdb_xy'] = self.robot.GetTrueLocalBaseVelocity()[0:2]
        info['fdb_yaw'] = self.robot.GetTrueBaseRollPitchYawRate()[2]
    
        if self.load_cmd.tracking:
            cmd_xy, cmd_yaw = self.load_cmd.get_target()
            info['cmd_xy'] = cmd_xy
            info['cmd_yaw'] = cmd_yaw
            info['cmd_scale'] = self.load_cmd.scale
            err_xy = info['cmd_scale'] * info['cmd_xy']- info['fdb_xy']
            err_yaw = info['cmd_scale'] * info['cmd_yaw']- info['fdb_yaw']
            obs = np.concatenate((cmd_xy, cmd_yaw, obs), axis=None)
        else:
            err_xy = 0
            err_yaw = 0
            obs = np.concatenate((self.load_cmd.get_target(), obs), axis=None)
            # here 'self.load_cmd.get_target()' returns 3D tuple of xy-axis velocity and yaw angular velocity commands, but y-axis velocity command is always zero. Therefore, it works as 2D tuple with 1D dummy value.

        info['Error XY'] = np.linalg.norm(err_xy)
        info['Error Yaw'] = np.linalg.norm(err_yaw)
        info['Drift'] = np.absolute(info['fdb_xy'][1] / np.max((info['fdb_xy'][0], 0.01)))
        info['Done'] = done

        info["Joint Power"] = self.robot.GetEnergyConsumptionPerControlStep()
        info["Contact"] = np.sum(self.robot.GetFootContacts())
        info["Height"] = self.robot.GetBasePosition()[2]
        info["Roll Rate"] = np.absolute(self.robot.GetBaseRollPitchYawRate()[0])
        info["Pitch Rate"] = np.absolute(self.robot.GetBaseRollPitchYawRate()[1])
        info["Yaw Rate"] = np.absolute(self.robot.GetBaseRollPitchYawRate()[2])

        return obs, rew, done, info


    def close(self):
        if self.load_terrain !=None:
            self.pybullet_client.removeBody(self.modelTerrain)
            del self.modelTerrain
        self.env.close()


class ObsHistoryWrapper(gym.Wrapper):

    def __init__(self, env, num_history):
        gym.Wrapper.__init__(self, env) 
        self.pybullet_client = self.env.pybullet_client
        self.num_history = num_history
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=np.tile(np.concatenate((self.env.observation_space.low, self.env.action_space.low), axis=None), self.num_history),
                                                high=np.tile(np.concatenate((self.env.observation_space.high, self.env.action_space.high), axis=None), self.num_history),
                                                shape=((self.env.observation_space.shape[0] + self.env.action_space.shape[0]) * self.num_history,), dtype=np.float32)
        self.sensor_space = self.env.observation_space
        self.robot = self.env.robot
        self.history_buffer = collections.deque(maxlen=self.num_history)


    def reset(self, **kwargs):
        obs,info = self.env.reset(**kwargs)
        self.robot = self.env.robot
        action = np.zeros(self.env.action_space.shape[0])

        for _ in range(self.num_history):
            self.history_buffer.appendleft(np.concatenate((obs, action), axis=None))

        self.prv_action = action
        
        return np.concatenate(self.history_buffer), info


    def step(self, action, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        self.history_buffer.appendleft(np.concatenate((obs, self.prv_action), axis=None))
        self.prv_action = action
        return np.concatenate(self.history_buffer), rew, done, info


class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, sensor_mode):
        gym.Wrapper.__init__(self, env) 
        self.robot = self.env.robot
        self.pybullet_client = self.env.pybullet_client
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.sensor_mode = sensor_mode if sensor_mode is not None else {} 

        if "force_vec" in self.sensor_mode.keys() and sensor_mode["force_vec"]:
            sensor_shape = self.observation_space.high.shape[0]
            obs_h = np.array([1]*(sensor_shape+6))
            obs_l = np.array([0]*(sensor_shape+6))
            self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)

        if "dynamic_vec" in self.sensor_mode.keys() and sensor_mode["dynamic_vec"]:
            sensor_shape = self.observation_space.high.shape[0]
            obs_h = np.array([1]*(sensor_shape+3))
            obs_l = np.array([0]*(sensor_shape+3))
            self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.robot = self.env.robot
        self.dynamic_info = info["dynamics"]

        if "force_vec" in self.sensor_mode.keys() and self.sensor_mode["force_vec"]:
            force_vec = info["force_vec"]
            obs = np.concatenate((obs,force_vec),axis = 0)

        if "dynamic_vec" in self.sensor_mode.keys() and self.sensor_mode["dynamic_vec"]:
            dynamic_vec = self.dynamic_info
            obs = np.concatenate((obs,dynamic_vec),axis = 0)

        return obs, info

    def step(self, action, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        
        if "force_vec" in self.sensor_mode.keys() and self.sensor_mode["force_vec"]:
            force_vec = info["force_vec"]
            obs = np.concatenate((obs,force_vec),axis = 0)

        if "dynamic_vec" in self.sensor_mode.keys() and self.sensor_mode["dynamic_vec"]:
            dynamic_vec = self.dynamic_info
            obs = np.concatenate((obs,dynamic_vec),axis = 0)

        return obs, rew, done, info


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_param, num_action_repeat):
        gym.Wrapper.__init__(self, env)  
        self.reward_param = copy.deepcopy(REWARD_PARAM_DICT)
        self.last_base10 = np.zeros((10,3))
        self.pybullet_client = self.env.pybullet_client
        self.num_action_repeat = num_action_repeat
        self.observation_space = self.env.observation_space
        self.sensor_space = self.env.sensor_space
        self.action_space = self.env.action_space
        self.steps = 0
        self.info = {}

        self.robot = self.env.robot

        for key in reward_param.keys():
            self.reward_param[key] = reward_param[key]


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.robot = self.env.robot
        self.steps = 0
        self.last_basepose = copy.copy(self.robot.GetBasePosition())
        self.last_baseorientation = copy.copy(self.robot.GetBaseOrientation())
        self.last_baseyaw = copy.copy(self.robot.GetBaseRollPitchYaw()[-1])
        self.last_footposition = copy.copy(self.get_foot_world())
        base_pose = copy.copy(self.robot.GetBasePosition())
        self.last_base10 = np.tile(base_pose,(10,1))
        self.info['dt'] = 0.002*self.num_action_repeat
        self.update_state(info)

        return obs, info


    def update_state(self, info):
        self.info['cur_xyz'] = self.robot.GetBasePosition()
        self.info['cur_rpy'] = self.robot.GetBaseRollPitchYaw()
        self.info['cur_q'] = self.robot.GetBaseOrientation()
        self.info['cur_dyaw'] = self.info['cur_rpy'][-1] - self.last_baseyaw
        self.info['cur_dxyz'] = np.array(self.robot.GetBasePosition())-np.array(self.last_basepose)
        self.info['cur_vel'] = self.robot.TransformAngularVelocityToLocalFrame(self.info['cur_dxyz'], self.info['cur_q'])/self.info['dt']

        if self.load_cmd.tracking:
            self.info['des_dyaw'] = info['cmd_scale'] * info['cmd_yaw'] * self.info['dt']
            cos_d = np.cos(self.info['des_dyaw'])
            sin_d = np.sin(self.info['des_dyaw'])
            self.info['des_vel'] = np.zeros(3)
            self.info['des_vel'][0:2] = info['cmd_scale'] * np.array([[cos_d, -sin_d],[sin_d, cos_d]]) @ np.array(info['cmd_xy'])
            self.info['des_dxyz'] = self.info['des_vel'] * self.info['dt']

        self.info['lost_contact'] = np.sum(1.0-np.array(self.robot.GetFootContacts()))
        self.info['bad_contact'] = self.robot.GetBadFootContacts()
        self.info['good_contact'] = np.max(np.sum(self.robot.GetFootContacts()) - self.info['bad_contact'], 0)

        self.info['energy'] = self.robot.GetEnergyConsumptionPerControlStep()


    def step(self, action,**kwargs):
        self.steps+=1
        obs, rew, done, info = self.env.step(action, **kwargs)
        self.update_state(info)
        info = self.reward_shaping(info)
        rewards = 0
        done = self.terminate()
        if done:
            info["RewardFail"] = -1
        else:
            info["RewardFail"] = 0
        for key in self.reward_param.keys():
            if key in info.keys():
                rewards+= info[key]
        self.last_basepose = copy.copy(self.robot.GetBasePosition())
        self.last_baseorientation = copy.copy(self.robot.GetBaseOrientation())
        self.last_baseyaw = self.robot.GetBaseRollPitchYaw()[-1]
        self.last_base10[1:,:] = self.last_base10[:9,:]
        self.last_base10[0,:] = np.array(self.robot.GetBasePosition()).reshape(1,3)
        self.last_footposition = self.get_foot_world()

        return obs, rewards, done, info


    def reward_shaping(self, info):

        if 'cmd_xy' and 'cmd_yaw' in info.keys():
            rew_tracking = self.reward_tracking()
            vel_diff = np.linalg.norm(self.info['cur_vel']- self.info['des_vel'])
            k = 1-np.tanh(2.95*vel_diff**2)
            # k = 1-np.tanh(2*vel_diff**2)
        else:
            rew_tracking = 0
            k = 1

        if 'Reward Tracking' in self.reward_param.keys():
            info['Reward Tracking'] = self.reward_param['Reward Tracking']*rew_tracking
        if 'Reward Balance' in self.reward_param.keys():
            info['Reward Balance'] = (self.reward_param['Reward Balance'])*self.reward_balance()*k
        if 'Reward Gait' in self.reward_param.keys():
            info['Reward Gait'] = self.reward_param['Reward Gait']*self.reward_gait() # not used
        if 'Reward Energy' in self.reward_param.keys():
            info['Reward Energy'] = -self.reward_param['Reward Energy']*self.info['energy']*k
        if 'Reward Badfoot' in self.reward_param.keys():
            info['Reward Badfoot'] = -self.reward_param['Reward Badfoot']*self.info['bad_contact']
        if 'Reward Footcontact' in self.reward_param.keys():
            info['Reward Footcontact'] = self.reward_param['Reward Footcontact']*(-max(self.info['lost_contact']-1,0) + 0.*self.info['good_contact'])
        return info
    

    def draw_direction(self):
        des_yaw = 0
        if self.render:
            id = self.pybullet_client.addUserDebugLine(lineFromXYZ=[self.info['cur_xyz'][0],self.info['cur_xyz'][1],0.6],
                                                    lineToXYZ=[self.info['cur_xyz'][0]+np.cos(des_yaw),self.info['cur_xyz'][1]+np.sin(des_yaw),0.6],
                                                    lineColorRGB=[1,0,1],lineWidth=2)
        return id


    def terminate(self):
        rot_mat = self.pybullet_client.getMatrixFromQuaternion(self.info['cur_q'])
        base_std = np.sum(np.std(self.last_base10,axis=0))
        return rot_mat[-1]<0.5  or (base_std<=2e-4 and self.steps>=10)


    def reward_balance(self):
        roll = self.info['cur_rpy'][0]
        pitch = self.info['cur_rpy'][1]
        return 1-np.tanh(30*(roll**2+pitch**2))


    def reward_gait(self):
        swing_min = -0.25
        swing_max = -0.1 #0.1

        foot_contact = np.array(self.robot.GetFootContacts())
        foot_pos = np.array(self.robot.GetFootPositionsInBaseFrame())
        foot_hip = np.array(self.robot.GetHipPositionsInBaseFrame())

        stance_cost = -0*(foot_pos - foot_hip - HEIGHT_CONTACT)**2
        swing_cost = (np.clip((foot_pos - foot_hip)[:,2], swing_min, swing_max) - swing_min)
        pos_cost = 0.*np.sum(np.clip((foot_pos - foot_hip)[:,0], -0.1, 0))
        reward_out = (np.sum(foot_contact @ stance_cost) + np.sum((1-foot_contact) @ swing_cost) + pos_cost)/4
        return reward_out


    def get_foot_world(self):
        foot = np.array(self.robot.GetFootPositionsInBaseFrame()).transpose()
        rot_mat = np.array(self.pybullet_client.getMatrixFromQuaternion(self.last_baseorientation)).reshape(-1,3)
        base = np.array(self.last_basepose).reshape(-1,1)
        foot_world = rot_mat.dot(foot)+base
        return foot_world.transpose()


    def reward_tracking(self):
        v_diff = np.linalg.norm(self.info['cur_vel']-self.info['des_vel'])
        v_reward = np.exp(-5*v_diff)
        yaw_diff=self.info['cur_dyaw']-self.info['des_dyaw']
        yaw_diff=(yaw_diff+np.pi)%(2*np.pi)-np.pi
        dyaw_diff=yaw_diff/self.info['dt']
        k = 1-np.tanh(2.95*dyaw_diff**2)
        # k = 1-np.tanh(2*dyaw_diff**2)
        return k*v_reward


class DynamicsWrapper(gym.Wrapper):
    def __init__(self, env, random_param, dynamic_param):
        gym.Wrapper.__init__(self, env)  
        self.random_param = random_param if random_param is not None else {} 
        self.dynamic_param = dynamic_param if dynamic_param is not None else {}
        self.pybullet_client = self.env.pybullet_client
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.robot = self.env.robot
        self.render = self.env.rendering_enabled


    def generate_randomforce(self):
        force_position = (np.random.random(3)-0.5)*2*np.array([0.2,0.05,0.05])
        force_vec = np.random.uniform(low=-1,high=1,size=(3,))*np.array([0.5,1,0.05])
        # force_vec = force_vec/np.linalg.norm(force_vec)*np.random.uniform(20,50)
        force_vec = 0.5*force_vec/np.linalg.norm(force_vec)*np.random.uniform(20,50)
        return force_position,force_vec

    def draw_forceline(self, force_position, force_vec):
        if self.render:
            self.pybullet_client.addUserDebugLine(lineFromXYZ=force_position,lineToXYZ=force_position+force_vec/50,
                                                    parentObjectUniqueId=self.robot.quadruped,
                                                    parentLinkIndex=-1,lineColorRGB=[1,0,0])


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.robot = self.env.robot

        info['dynamics'] ={}

        ratio_param = {}
        if "random_dynamics" in self.random_param.keys() and self.random_param["random_dynamics"]:
            ratio_param['control_latency'] = np.random.uniform(low=0.0, high=2.0, size=1)
            ratio_param['joint_friction'] = np.random.uniform(low=0.0, high=2.0, size=12)
            ratio_param['spin_foot_friction'] = np.random.uniform(low=0.0, high=2.0, size=1)
            ratio_param['foot_friction'] = np.random.uniform(0.5, 1.5, size=1)
            ratio_param['base_mass'] = np.random.uniform(0.8, 1.2, size=1)
            ratio_param['base_inertia'] = np.random.uniform(0.8, 1.2, size=3)
            ratio_param['leg_mass'] = np.random.uniform(0.8, 1.2, size=3)
            ratio_param['leg_inertia'] = [np.random.uniform(0.8, 1.2, size=3) for _ in range(4)]
        else:
            for key in ['control_latency', 'spin_foot_friction', 'foot_friction', 'base_mass']:
                ratio_param[key] = 1
            for key in ['base_inertia', 'leg_mass']:
                ratio_param[key] = np.ones(3)
            ratio_param['joint_friction'] = np.ones(12)
            ratio_param['leg_inertia'] = [np.ones(3) for _ in range(4)] 

        if 'motor_kp' in self.dynamic_param.keys() and 'motor_kd' in self.dynamic_param.keys():
            motor_kp = self.dynamic_param['motor_kp']
            motor_kd = self.dynamic_param['motor_kd']
            self.robot.SetMotorGains(motor_kp, motor_kd)

        if 'control_latency' in self.dynamic_param.keys():
            control_latency = ratio_param['control_latency'] * self.dynamic_param['control_latency']
            self.robot.SetControlLatency(control_latency)
            info['dynamics']['control_latency'] = control_latency

        if 'joint_friction' in self.dynamic_param.keys():
            joint_friction = ratio_param['joint_friction'] * self.dynamic_param['joint_friction']
            self.robot.SetJointFriction(joint_friction)
            info['dynamics']['joint_friction'] = joint_friction

        if 'spin_foot_friction' in self.dynamic_param.keys():
            spin_foot_friction = ratio_param['spin_foot_friction'] * self.dynamic_param['spin_foot_friction']
            self.robot.SetFootSpinFriction(spin_foot_friction)
            info['dynamics']['spin_foot_friction'] = spin_foot_friction

        if 'foot_friction' in self.dynamic_param.keys():
            foot_friction = ratio_param['foot_friction'] * self.dynamic_param['foot_friction']
            self.robot.SetFootFriction(foot_friction)
            info['dynamics']['foot_friction'] = foot_friction

        if 'base_mass' in self.dynamic_param.keys():
            base_mass = ratio_param['base_mass'] * self.dynamic_param['base_mass']
        else:
            base_mass = ratio_param['base_mass'] * self.robot.GetBaseMassesFromURDF()[0]
        self.robot.SetBaseMasses([base_mass])
        info['dynamics']['base_mass'] = base_mass

        if 'base_inertia' in self.dynamic_param.keys():
            base_inertia = ratio_param['base_inertia'] * self.dynamic_param['base_inertia']
        else:
            base_inertia = ratio_param['base_inertia'] * self.robot.GetBaseInertiasFromURDF()
        self.robot.SetBaseInertias([base_inertia])
        info['dynamics']['base_inertia'] = base_inertia

        if 'leg_mass' in self.dynamic_param.keys():
            leg_mass = np.tile(ratio_param['leg_mass'], 4) * self.dynamic_param['leg_mass']
        else:
            leg_mass = np.tile(ratio_param['leg_mass'], 4) * self.robot.GetLegMassesFromURDF()
        self.robot.SetLegMasses(leg_mass)
        info['dynamics']['leg_mass'] = leg_mass

        leg_inertia_ratio = ratio_param['leg_inertia']
        if 'leg_inertia' in self.dynamic_param.keys():
            leg_inertia_base = self.dynamic_param['leg_inertia'] * 4
        else:
            leg_inertia_base = self.robot.GetLegInertiasFromURDF()
        leg_inertia = []
        for i in range(12):
            leg_inertia.append(leg_inertia_ratio[i%3]*leg_inertia_base[i])
        self.robot.SetLegInertias(leg_inertia)
        info['dynamics']['leg_inertia'] = leg_inertia

        if 'gravity' in self.dynamic_param.keys():
            gravity = self.dynamic_param['gravity']
            self.pybullet_client.setGravity(*gravity)
            info['dynamics']['gravity'] = gravity

        force_info = np.zeros(6)
        if "random_force" in self.random_param.keys() and self.random_param["random_force"]:
            self.pybullet_client.removeAllUserDebugItems()
            self.force_position, self.force_vec = self.generate_randomforce()
            self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped,linkIndex=-1,forceObj=self.force_vec,
                                            posObj=self.force_position,flags=self.pybullet_client.LINK_FRAME)
            self.draw_forceline(self.force_position,self.force_vec)
            force_info = np.concatenate((self.force_position/np.array([0.2,0.05,0.05]),self.force_vec/50),axis=0)
        info['force_vec'] = force_info

        return obs, info


    def step(self, action, **kwargs):

        force_info = np.zeros(6)
        obs, rew, done, info = self.env.step(action, **kwargs)
        if "random_force" in self.random_param.keys() and self.random_param["random_force"]:
            # New random force
            if self.env.env_step_counter % 100 == 0:
                self.force_position, self.force_vec = self.generate_randomforce()
                self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped,linkIndex=-1,forceObj=self.force_vec,
                                    posObj=self.force_position,flags=self.pybullet_client.LINK_FRAME)
                self.draw_forceline(self.force_position,self.force_vec)
                force_info = np.concatenate((self.force_position/np.array([0.2,0.05,0.05]),self.force_vec/50),axis=0)
            # Apply force
            elif self.env.env_step_counter % 100 <10:
                self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped,linkIndex=-1,forceObj=self.force_vec,
                                    posObj=self.force_position,flags=self.pybullet_client.LINK_FRAME)
                self.draw_forceline(self.force_position,self.force_vec)
                force_info = np.concatenate((self.force_position/np.array([0.2,0.05,0.05]),self.force_vec/50),axis=0)
            # delete line
            elif self.env.env_step_counter % 100 ==10:
                self.pybullet_client.removeAllUserDebugItems() 
        info['force_vec'] = force_info 
        return obs, rew, done, info
