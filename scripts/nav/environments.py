import os

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as c
import pybullet_data
import copy
import pickle

import gym

import a1sim
from a1sim.envs.env_builder import SENSOR_MODE

from gait.wrapper import TianshouWrapper, TrajectoryHandler

REWARD_PARAM_DICT = {'Reward Tracking':0, 'Reward Balance':0, 'Reward Gait':0, 'Reward Energy':0, 'Reward Fail':0, 'Reward Badfoot':0, 'Reward Footcontact':0}
RANDOM_PARAM_DICT = {'random_dynamics':0, 'random_force':0}
DYNAMIC_PARAM_DICT = {'control_latency': 0.001, 'joint_friction': 0.025, 'spin_foot_friction': 0.2, 'foot_friction': 1}

DEG_TO_RAD = np.pi/180.
RAD_TO_DEG = 180./np.pi

SIM_INTERVAL = 0.002

FIELD_RANGE = [50, 3]
FURNITURES = {}
OBJECTS = {}

for file in os.listdir('data/assets/furnitures'):
    if os.path.isdir(os.path.join('data', 'assets', 'furnitures', file)):
        FURNITURES[file] = {'urdfPath': os.path.join('assets', 'furnitures', file, 'model.urdf'), 'basePosition':np.zeros(3), 'baseOrientation': np.zeros(3), 'scale':1}

for file in os.listdir('data/assets/objects'):
    if os.path.isdir(os.path.join('data', 'assets', 'objects', file)):
        OBJECTS[file] = {'urdfPath': os.path.join('assets', 'objects', file, 'model.urdf'), 'basePosition':np.zeros(3), 'baseOrientation': np.zeros(3), 'scale':1}


class PerceptionWrapper(gym.Wrapper):
    def __init__(self, env, 
                 loco_command, loco_agent, 
                 time_step = 0.1, time_max=200., env_type=2, record=None):
        gym.Wrapper.__init__(self, env)

        self._view_agent = {'dist': 0.2,
                            'offset': np.array([0.45, 0, 0.]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -90. * DEG_TO_RAD,
                            'pitch': 0. * DEG_TO_RAD,
                            'width': 212,
                            'height': 120,
                            'near': 0.1,
                            'far': 100}

        self._view_bird = {'dist': 2.0,
                            'offset': np.array([0.45, 0, 0]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -60. * DEG_TO_RAD,
                            'pitch': -30 * DEG_TO_RAD,
                            'width': 480,
                            'height': 360,
                            'near': 0.1,
                            'far': 100}

        self._view_fpv = {'dist': 1.5,
                            'offset': np.array([0.45, 0, 0]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -90. * DEG_TO_RAD,
                            'pitch': -30 * DEG_TO_RAD,
                            'width': 480,
                            'height': 360,
                            'near': 0.1,
                            'far': 100}

        image_size = (self._view_agent['width'], self._view_agent['height'], 4)

        self.gait_action_space = self.env.action_space
        self.gait_observation_space = self.env.observation_space
        self.gait_sensor_space = self.env.sensor_space
        self.nav_action_space = gym.spaces.Box(low=-1, high=1,
                                              shape=(2,), dtype=np.float32)
        self.nav_observation_space = gym.spaces.Box(low=np.array([0] * np.prod(image_size)),
                                                    high=np.array([1] * np.prod(image_size)),
                                                    shape=(np.prod(image_size),), dtype=np.float32)

        self.time_max = time_max
        self.time_step = time_step
        self.loco_command = loco_command
        self.loco_agent = loco_agent

        self.env_type = env_type
        self._record = record

        self._gait_action_init = np.zeros(12)
        self._nav_action_init = np.zeros(2)


    def seed(self, seed):
        self.env.seed(seed)
        pass        

    def reset(self, env_profile=None, **kwargs):

        self.time_cur = 0.
        self._nav_action = self._nav_action_init
        self._gait_action = self._gait_action_init

        kwargs['hard_reset'] = True

        # Simulation initiation
        self._gait_obs = self.env.reset(**kwargs)
        self._yaw = 0.0

        self.loco_agent.reset()

        if env_profile == None:
            if self.env_type == None:
                env_type = np.random.choice(5, p=[0.17, 0.17, 0.32, 0.17, 0.17]) 
            else:
                env_type = self.env_type

            if env_type==-1:
                self.env_profile = self._generate_obstacle_profiles(numPeople = 0, 
                                                                    numObjects=0,
                                                                    numFurnitures=0,
                                                                    lenUnit=45)
            elif env_type==4:
                self.env_profile = self._generate_obstacle_profiles(numPeople = 3, 
                                                                    numObjects=0,
                                                                    numFurnitures=0,
                                                                    lenUnit=20)
            elif env_type==3:
                self.env_profile = self._generate_obstacle_profiles(numPeople = 0, 
                                                                    numObjects=4,
                                                                    numFurnitures=4,
                                                                    lenUnit=30)
            elif env_type==2:
                self.env_profile = self._generate_obstacle_profiles(numPeople = 1, 
                                                                    numObjects=2,
                                                                    numFurnitures=2,
                                                                    lenUnit=45)
            elif env_type==1:
                self.env_profile = self._generate_obstacle_profiles(numPeople = 1, 
                                                                    numObjects=3,
                                                                    numFurnitures=3,
                                                                    lenUnit=20)
            else:
                self.env_profile = self._generate_obstacle_profiles(numPeople = 1, 
                                                                    numObjects=3,
                                                                    numFurnitures=3,
                                                                    lenUnit=45)
        else:
            self.env_profile = copy.copy(env_profile)

        self.assets = self._set_world(self.env_profile)
        self._apply_action()
        obs = self._get_observation()

        self.obstacles_to_go = []
        for person in self.assets['people']:
            self.obstacles_to_go.append(person.id)
        for obstacle_id in self.assets['furnitures'] + self.assets['objects']:
            self.obstacles_to_go.append(obstacle_id)

        self._distance = 0.0
        self._success = 0
        self.stay_cnt = 0
        self.stay_time = 5

        self.time_cur = self.env.get_time_since_reset()
        self.time_next_rec = 0.0
        self.time_step_rec = 1./30

        self.img_list = []

        return obs
    

    def step(self, action):

        self._nav_action = np.clip(action, [0, -1.0], [1., 1.0])
        self._apply_action()

        while self.env.get_time_since_reset() < self.time_cur + self.time_step:
            gait_done, info = self._step_simulation()
            
            if self._record is not None:
                if self.env.get_time_since_reset() >= self.time_next_rec:
                    self.img_list.append(self.render(self._record))
                    self.time_next_rec += self.time_step_rec
                
        self.time_cur += self.time_step

        obs = self._get_observation()
        rew, rew_info = self._get_reward()
        nav_done, term_info = self._check_termination()

        done = gait_done or nav_done

        for key, value in rew_info.items():
            info[key] = value

        if done:
            for key, value in term_info.items():
                info[key] = value

            if term_info['Success']:
                info['Fail'] = 0
                self._success = 1
            else:
                info['Fail'] = 1
                self._success = 0

            info['done'] = 1

        return obs, rew, done, info


    def close(self):
        self.env.close()


    def _apply_action(self):

        self.loco_command.set_target(self._nav_action)


    def _update_people(self):
        for person in self.assets['people']:
            position_person, _ = self.env.pybullet_client.getBasePositionAndOrientation(person.id)
            position_agent, _ = self.env.pybullet_client.getBasePositionAndOrientation(self.assets['agent'])
            if position_agent[0] + 0.5 > position_person[0]:
                person.deactive()
            person.step()


    def _get_reward(self):

        rew_perception = 0
        rew_distance = 0

        position_agent, _ = self.env.pybullet_client.getBasePositionAndOrientation(self.assets['agent'])

        for obstacle_id in self.obstacles_to_go:

            position_obstacle, _ = self.env.pybullet_client.getBasePositionAndOrientation(obstacle_id)

            if position_agent[0] > position_obstacle[0]:
                self.obstacles_to_go.remove(obstacle_id)
                rew_perception += 200

        if position_agent[0] > self._distance:
            rew_distance += 10 * (position_agent[0] - self._distance)
            self._distance = position_agent[0]
            self.stay_cnt = 0
        else:
            self.stay_cnt +=1

        rew = rew_perception + rew_distance

        return rew,  {'Reward Perception': rew_perception, 'Reward Distance': rew_distance}


    def _get_observation(self):

        position, orientation = self.env.pybullet_client.getBasePositionAndOrientation(self.assets['agent'])
        view_point, _ = self.env.pybullet_client.multiplyTransforms(position, orientation, self._view_agent['offset'], [0, 0, 0, 1])
        view_rpy = self.env.pybullet_client.getEulerFromQuaternion(orientation)

        view_matrix = self.env.pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition = view_point,
            distance = self._view_agent['dist'],
            roll = RAD_TO_DEG * (view_rpy[0] + self._view_agent['roll']),
            pitch = RAD_TO_DEG * (view_rpy[1] + self._view_agent['pitch']),
            yaw = RAD_TO_DEG * (view_rpy[2] + self._view_agent['yaw']),
            upAxisIndex=2)
        proj_matrix = self.env.pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._view_agent['width']) / self._view_agent['height'],
            nearVal=self._view_agent['near'],
            farVal=self._view_agent['far'])
        (_, _, rgb, depth, _) = self.env.pybullet_client.getCameraImage(
            width=self._view_agent['width'],
            height=self._view_agent['height'],
            renderer=self.env.pybullet_client.ER_TINY_RENDERER,
            viewMatrix=view_matrix,
            shadow=0,
            projectionMatrix=proj_matrix)

        self._pixels = {}
        self._pixels['rgb'] = np.array(rgb)[:, :, 2::-1]
        self._pixels['depth'] = np.array((1-depth)*255, dtype=np.uint8)
        rgbd = np.concatenate((self._pixels['rgb'], np.sqrt(self._pixels['depth'])[:, :, np.newaxis]), axis=2)/255.
        # rgbd = np.concatenate((self._pixels['rgb'], np.sqrt(self._pixels['depth'])[:, :, np.newaxis]), axis=2)/255.

        self.env.pybullet_client.resetDebugVisualizerCamera( cameraDistance = self._view_fpv['dist'],
                                                             cameraTargetPosition = view_point,
                                                             cameraPitch = RAD_TO_DEG * self._view_fpv['pitch'],
                                                             cameraYaw = RAD_TO_DEG * self._view_fpv['yaw']
                                                             )

        self._yaw = view_rpy[2]

        obs = {'rgbd': rgbd, 'yaw':self._yaw, 'action': self._nav_action}

        return obs


    def _step_simulation(self):

        self._update_people()
        self._gait_action = self.loco_agent.predict(self._gait_obs)
        self._gait_obs, _, done, info = self.env.step(self._gait_action)

        return done, info


    def _check_termination(self):

        collision = False
        timeout = False
        goal = False

        contact_points = self.env.pybullet_client.getContactPoints(bodyA=self.assets['agent'])
        for point in contact_points:
            if point[2] == self.assets['agent'] or point[2] == self.assets['ground']:
                continue
            else:
                collision = True

        position, orientation = self.env.pybullet_client.getBasePositionAndOrientation(self.assets['agent'])
        goal = (position[0] >= FIELD_RANGE[0])
        sidewalk = (position[1] >= 0.5*FIELD_RANGE[1]) or (position[1] <= -0.5*FIELD_RANGE[1])

        timeout = (self.time_cur > self.time_max)
        stay = self.stay_cnt * self.time_step > self.time_max

        return collision or timeout or sidewalk or stay or goal, {'Collision': 1*collision, 'Timeout':1*timeout, 'Sidewalk':1*sidewalk, 'Stay':1*stay, 'Success':1*goal} 


    def render(self, key=None):

        if key == 'rgb':
            return self._pixels['rgb']
        elif key == 'depth':
            return self._pixels['depth']
        elif key == 'fpv':
            position, orientation = self.env.pybullet_client.getBasePositionAndOrientation(self.assets['agent'])
            view_point, _ = self.env.pybullet_client.multiplyTransforms(position, orientation, self._view_agent['offset'], [0, 0, 0, 1])
            view_rpy = self.env.pybullet_client.getEulerFromQuaternion(orientation)
            view_matrix = self.env.pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition = view_point,
                distance = self._view_fpv['dist'],
                roll = RAD_TO_DEG * self._view_fpv['roll'],
                pitch = RAD_TO_DEG * self._view_fpv['pitch'],
                yaw = RAD_TO_DEG * (view_rpy[2] + self._view_fpv['yaw']),
                upAxisIndex=2)
            proj_matrix = self.env.pybullet_client.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self._view_fpv['width']) / self._view_fpv['height'],
                nearVal=self._view_fpv['near'],
                farVal=self._view_fpv['far'])
            (_, _, rgb, depth, _) = self.env.pybullet_client.getCameraImage(
                width=self._view_fpv['width'],
                height=self._view_fpv['height'],
                renderer=self.env.pybullet_client.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                shadow=0,
                projectionMatrix=proj_matrix)
        else:
            position, orientation = self.env.pybullet_client.getBasePositionAndOrientation(self.assets['agent'])
            position = np.array([position[0], 0.0, 0.25])
            view_point, _ = self.env.pybullet_client.multiplyTransforms(position, orientation, self._view_agent['offset'], [0, 0, 0, 1])
            view_rpy = self.env.pybullet_client.getEulerFromQuaternion(orientation)
            view_matrix = self.env.pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition = view_point,
                distance = self._view_bird['dist'],
                roll = RAD_TO_DEG * self._view_bird['roll'],
                pitch = RAD_TO_DEG * self._view_bird['pitch'],
                yaw = RAD_TO_DEG * self._view_bird['yaw'],
                upAxisIndex=2)
            proj_matrix = self.env.pybullet_client.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self._view_bird['width']) / self._view_bird['height'],
                nearVal=self._view_bird['near'],
                farVal=self._view_bird['far'])
            (_, _, rgb, depth, _) = self.env.pybullet_client.getCameraImage(
                width=self._view_bird['width'],
                height=self._view_bird['height'],
                renderer=self.env.pybullet_client.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                shadow=0,
                projectionMatrix=proj_matrix)

        img = np.array(rgb)[:, :, 2::-1]

        return img


    def record(self):
        out = copy.copy(self.img_list)
        self.img_list = []
        return out


    def _set_world(self, setup_obstacles):

        assets = {}

        assets['ground'] = self.env.get_ground()

        mass = 0

        idColWall = self.env.pybullet_client.createCollisionShape(self.env.pybullet_client.GEOM_PLANE)
        assets['walls'] = [ self.env.pybullet_client.createMultiBody(mass, idColWall, -1, [0.5 * FIELD_RANGE[0],    + 0.5*FIELD_RANGE[1] + 0.2, 1.0], [ 0.707, 0, 0,  0.707]),
                            self.env.pybullet_client.createMultiBody(mass, idColWall, -1, [0.5 * FIELD_RANGE[0],    - 0.5*FIELD_RANGE[1] - 0.2, 1.0], [ 0.707, 0, 0, -0.707]),
                            self.env.pybullet_client.createMultiBody(mass, idColWall, -1, [0.0,                       0.0,                      2.0], [     1, 0, 0,  0])]

        self.env.pybullet_client.changeVisualShape(assets['ground'], -1, textureUniqueId=self.env.pybullet_client.loadTexture("textures/carpet.png"))
        self.env.pybullet_client.changeVisualShape(assets['walls'][0], -1, textureUniqueId=self.env.pybullet_client.loadTexture("textures/wall.png"), rgbaColor=(0.8, 0.7, 0.8, 1.0))
        self.env.pybullet_client.changeVisualShape(assets['walls'][1], -1, textureUniqueId=self.env.pybullet_client.loadTexture("textures/wall.png"), rgbaColor=(0.8, 0.7, 0.8, 1.0))
        self.env.pybullet_client.changeVisualShape(assets['walls'][2], -1, textureUniqueId=self.env.pybullet_client.loadTexture("textures/wall.png"), rgbaColor=(0.6, 0.7, 0.9, 1.0))

        assets['people'] = []
        assets['furnitures'] = []
        assets['objects'] = []

        for dict_item in setup_obstacles['furnitures']:

            object = dict_item['choice']
            basePosition = dict_item['position']
            baseOrientation = dict_item['orientation']
            scale = object['scale']
            path = object['urdfPath']
            assets['furnitures'].append(self.env.pybullet_client.loadURDF(path, basePosition, baseOrientation, globalScaling = scale, useFixedBase = True))

        for dict_item in setup_obstacles['objects']:

            object = dict_item['choice']
            basePosition = dict_item['position']
            baseOrientation = dict_item['orientation']
            scale = object['scale']
            path = object['urdfPath']
            assets['objects'].append(self.env.pybullet_client.loadURDF(path, basePosition, baseOrientation, globalScaling = scale, useFixedBase = False))

        for dict_item in setup_obstacles['people']:
            basePosition = dict_item['position']
            baseOrientation = dict_item['orientation']
            scale = dict_item['scale']
            assets['people'].append(ObjectHuman(self.env.pybullet_client, basePosition, baseOrientation, scale))

        assets['agent'] = self.env.robot.quadruped

        return assets


    def _generate_obstacle_profiles(self, numPeople, numObjects, numFurnitures, lenUnit=50, lenOffset=2):

        setup_obstacles = {}

        setup_obstacles['furnitures'] = []
        setup_obstacles['objects'] = []
        setup_obstacles['people'] = []

        for cntUnit in range(int((FIELD_RANGE[0]-lenOffset)/lenUnit)):

            raRandPos = np.random.uniform(low=0, high=1, size=(numFurnitures, 2))
            valRandYaw = np.random.uniform(low=0, high=1)

            for idx in range(numFurnitures):

                dict_item = {}
                dict_item['choice'] = copy.copy(np.random.choice(list(FURNITURES.values())))
                basePosition = np.array([lenOffset + lenUnit * (cntUnit + raRandPos[idx,0]), 
                                        FIELD_RANGE[1] *(raRandPos[idx,1] - 0.5),
                                        0.]) + dict_item['choice']['basePosition']
                dict_item['position'] = basePosition
                dict_item['orientation'] = self.env.pybullet_client.getQuaternionFromEuler(
                                            dict_item['choice']['baseOrientation'] + valRandYaw * np.array([0, 0, 2*np.pi]))

                setup_obstacles['furnitures'].append(copy.copy(dict_item))

                numClusterObjects = np.random.randint(low=0, high=3)
                for idx in range(numClusterObjects):
                    dict_item = {}

                    dict_item['choice'] = copy.copy(np.random.choice(list(OBJECTS.values())))
                    valRandZ = np.random.uniform(low=0, high=1)
                    valRandYaw = np.random.uniform(low=0, high=1)
                    dict_item['position'] = basePosition + np.array([0, 0, valRandZ])
                    dict_item['orientation'] = self.env.pybullet_client.getQuaternionFromEuler(
                                                dict_item['choice']['baseOrientation'] + valRandYaw * np.array([0, 0, 2*np.pi]))
                    setup_obstacles['objects'].append(copy.copy(dict_item))

            raRandPos = np.random.uniform(low=0, high=1, size=(numObjects, 2))
            valRandYaw = np.random.uniform(low=0, high=1)

            for idx in range(numObjects):
                dict_item['choice'] = copy.copy(np.random.choice(list(OBJECTS.values())))
                dict_item['position'] = np.array([lenOffset + lenUnit * (cntUnit + raRandPos[idx,0]), 
                                        FIELD_RANGE[1] *(raRandPos[idx,1] - 0.5),
                                        0.]) + dict_item['choice']['basePosition']
                dict_item['orientation'] = self.env.pybullet_client.getQuaternionFromEuler(
                                        dict_item['choice']['baseOrientation'] + valRandYaw * np.array([0, 0, 2*np.pi]))
                setup_obstacles['objects'].append(copy.copy(dict_item))


            raRandPos = np.random.uniform(low=0, high=1, size=(numPeople, 2))        
            valRandYaw = np.random.uniform(low=0, high=1, size=numPeople)
            valRandScale = np.random.uniform(low=0, high=1, size=numPeople)

            for idx in range(numPeople):
                dict_item = {}
                dict_item['position'] =  np.array([lenOffset + lenUnit * (cntUnit + raRandPos[idx,0]), 
                                        FIELD_RANGE[1] *(raRandPos[idx,1] - 0.5),
                                        0.])
                dict_item['orientation'] = self.env.pybullet_client.getQuaternionFromEuler(valRandYaw[idx] * np.array([0, 0, 2*np.pi]))
                dict_item['scale'] = 0.4 * valRandScale[idx] + 0.6
                setup_obstacles['people'].append(copy.copy(dict_item))

        return setup_obstacles


    def save_obstacle_profiles(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.env_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def episode_eval_stat(self):
        """
        Return:
           distance: distance traveled
        """
        
        return {
            "Distance": self._distance,
            "Success": self._success
        }


class ObjectHuman():

    def __init__(self, pybullet_client, basePosition, baseOrientation, scale=1) -> None:
        self.client = pybullet_client
        self.idHuman = self.client.loadURDF("assets/human/human.urdf", 
                                             basePosition, baseOrientation, 
                                             globalScaling = scale, useFixedBase = False)

        dicJointNameToID = {}

        for idJoint in range(self.client.getNumJoints(self.idHuman)):
            jointInfo = self.client.getJointInfo(self.idHuman, idJoint)
            dicJointNameToID[jointInfo[1].decode('UTF-8')] = jointInfo[0]

        self.joint_left_elbow = dicJointNameToID["left_elbow"]
        self.joint_left_shoulder = dicJointNameToID["left_shoulder"]

        self.joint_right_elbow = dicJointNameToID["right_elbow"]
        self.joint_right_shoulder = dicJointNameToID["right_shoulder"]

        self.joint_left_hip = dicJointNameToID["left_hip"]
        self.joint_left_knee = dicJointNameToID["left_knee"]
        self.joint_left_ankle = dicJointNameToID["left_ankle"]

        self.joint_right_hip = dicJointNameToID["right_hip"]
        self.joint_right_knee = dicJointNameToID["right_knee"]
        self.joint_right_ankle = dicJointNameToID["right_ankle"]

        self.joint_chest = dicJointNameToID["chest"]
        self.joint_neck = dicJointNameToID["neck"]

        self.phase = 0
        self.time_step = 0.002 * 10
        self.frequency = 1
        self.magnitude = 1

        self.cmd_lin = 0.4
        self.cmd_yaw = 0
        self.active = True

    @property
    def id(self):
        return self.idHuman

    def deactive(self):
        self.active = False

    def step(self):

        self.client.setJointMotorControlMultiDof(self.idHuman, self.joint_neck, self.client.POSITION_CONTROL, 
                                          targetPosition = [0,0,0])
        self.client.setJointMotorControlMultiDof(self.idHuman, self.joint_chest, self.client.POSITION_CONTROL, 
                                          targetPosition = [0,0,0])

        self.client.setJointMotorControl2(self.idHuman, self.joint_left_elbow, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * 1. * self.magnitude)
        self.client.setJointMotorControl2(self.idHuman, self.joint_left_shoulder, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * (-0.5) *self.magnitude)

        self.client.setJointMotorControl2(self.idHuman, self.joint_right_elbow, self.client.POSITION_CONTROL, 
                                          targetPosition = np.sin(2 * np.pi * self.frequency * self.phase) * (-1.) * self.magnitude)
        self.client.setJointMotorControl2(self.idHuman, self.joint_right_shoulder, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * 0.5 * self.magnitude)

        self.client.setJointMotorControl2(self.idHuman, self.joint_left_hip, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * 0.5 * self.magnitude)
        self.client.setJointMotorControl2(self.idHuman, self.joint_left_knee, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * (-0.5) * self.magnitude)                                         
        self.client.setJointMotorControl2(self.idHuman, self.joint_left_ankle, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * 0.25 * self.magnitude)

        self.client.setJointMotorControl2(self.idHuman, self.joint_right_hip, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * (-0.5) * self.magnitude)
        self.client.setJointMotorControl2(self.idHuman, self.joint_right_knee, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * 0.5 * self.magnitude)                                          
        self.client.setJointMotorControl2(self.idHuman, self.joint_right_ankle, self.client.POSITION_CONTROL, 
                                          targetPosition= np.sin(2 * np.pi * self.frequency * self.phase) * (-0.25) * self.magnitude)
        
        self.cmd_lin += 0.5*np.random.normal(0, 1) * self.time_step
        self.cmd_yaw = 0.5*np.random.normal(0, 1)* self.time_step

        self.cmd_lin = np.clip(self.cmd_lin, 0, 0.75)
        self.cmd_yaw = np.clip(self.cmd_yaw, -0.2, -0.2)

        position, orientation = self.client.getBasePositionAndOrientation(self.idHuman)
        rpy = self.client.getEulerFromQuaternion(orientation)
        if (( position[1] > 0.5 * FIELD_RANGE[1] - 0.25 and np.sin(rpy[2])>0) or
            (-position[1] > 0.5 * FIELD_RANGE[1] - 0.25 and np.sin(rpy[2])<0) or 
            ( position[0] > 1.0 * FIELD_RANGE[0] - 0.25 and np.cos(rpy[2])>0)):
            self.cmd_lin = 0
        if not self.active:
            self.cmd_lin = 0
            self.cmd_yaw = 0

        lin_vel, _ = self.client.multiplyTransforms([0, 0, 0], orientation, [self.cmd_lin, 0, 0], [0, 0, 0, 1])
        ang_vel = np.array([0, 0, self.cmd_yaw])
        self.client.resetBaseVelocity(self.idHuman, lin_vel, ang_vel)

        self.phase += self.cmd_lin * self.time_step



class DemoEnv(gym.Env):

    def __init__(self, cmd, render=False, time_sim = SIM_INTERVAL, time_step = SIM_INTERVAL*10, time_init = 0.5):

        self.action_space = gym.spaces.Box( low=-1,
                                        high=1,
                                        shape=(2,), dtype=np.float32)

        # Not using yet
        self.observation_space = gym.spaces.Box(low=np.array(  [-1.0, -1.0, -1.0]),
                                            high=np.array( [ 1.0,  1.0,  1.0]),
                                            shape=(3, ), dtype=np.float32)
        self.sensor_space = gym.spaces.Box(low=np.array(  [-1.0, -1.0, -1.0]),
                                            high=np.array( [ 1.0,  1.0,  1.0]),
                                            shape=(3, ), dtype=np.float32)

        self.load_cmd = cmd
        self.time_sim = time_sim
        self.time_step = time_step
        self.time_init = time_init
        self.flag_render = render

        if self.flag_render:
            self.pybullet_client = c.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
        else:
            self.pybullet_client = c.BulletClient(connection_mode=p.DIRECT)


    def seed(self):
        pass


    def reset(self, **kwargs):

        self.pybullet_client.resetSimulation()
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pybullet_client.setGravity(0, 0, -10)

        ## Set up simulation
        num_bullet_solver_iterations = 30
        self.pybullet_client.setTimeStep(self.time_sim)
        self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
        self.pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)

        self.ground = self.pybullet_client.loadURDF("plane.urdf")

        self.robot = DemoRobot(self.pybullet_client, self.time_sim)

        cmd_xy, cmd_yaw = self.load_cmd.reset()

        # Simulation initiation
        while self.robot.GetTimeSinceReset() < self.time_init:
            self.robot.Step(cmd_xy, cmd_yaw)
            self.pybullet_client.stepSimulation()

        self.time_cur = self.time_init

        # Observation initiation
        obs = 0
        info = {}

        return obs, info

    def get_ground(self):
        return self.ground


    def step(self, action):

        while self.robot.GetTimeSinceReset() < self.time_cur + self.time_step:
            cmd_xy, cmd_yaw = self.load_cmd.get_target()
            self.robot.Step(cmd_xy, cmd_yaw)
            self.pybullet_client.stepSimulation()

        self.time_cur += self.time_step

        obs = 0
        rew = 0
        done = False
        info = {}

        return obs, rew, done, info


    def close(self):

        self.pybullet_client.disconnect()

    def get_time_since_reset(self):
        return self.robot.GetTimeSinceReset()



class DemoRobot():
    def __init__(self, pybullet_client, time_step):

        self.client = pybullet_client
        mass = 0
        basePosition = [0., 0., 0.26]
        baseOrientation = [0, 0, 0, 1]
        idColAgent = self.client.createCollisionShape(self.client.GEOM_CYLINDER, radius=0.5, height=0.5)
        idVizAgent = self.client.createVisualShape(self.client.GEOM_BOX, halfExtents=[0.4, 0.3, 0.12])

        self.quadruped = self.client.createMultiBody(mass, idColAgent, idVizAgent, basePosition, baseOrientation)
        self.time_cur = 0.
        self.time_step = time_step


    def GetTimeSinceReset(self):
        return self.time_cur


    def Step(self, cmd_xy, cmd_yaw):

        position, orientation = self.client.getBasePositionAndOrientation(self.quadruped)
        lin_vel, _ = self.client.multiplyTransforms([0, 0, 0], orientation, [cmd_xy[0], cmd_xy[1], 0], [0, 0, 0, 1])
        ang_vel = np.array([0, 0, cmd_yaw])

        self.client.resetBaseVelocity(self.quadruped, lin_vel, ang_vel)
        self.time_cur += self.time_step
        

class DummyAgent():
    def __init__(self):
        self.action = np.zeros(12)
    def predict(self, obs):
        return self.action


def build_demo_env(render, env_type, record=None):

    cmd_trj = TrajectoryHandler()
    cmd_trj.set_scale = 1.
    env = DemoEnv(cmd=cmd_trj, render=render)
    dummy_agent = DummyAgent()

    return PerceptionWrapper(env = env,
                            loco_command = cmd_trj,                            
                            loco_agent = dummy_agent,
                            time_step = 0.1, time_max = 200.0,
                            env_type = env_type, record = record)


def build_robot_env(sim_config, gait_agent, rew_config={}, env_type=None, render=False, record=None):

    print(sim_config)
    # Configure envirnoment setup.
    reward_param = copy.copy(REWARD_PARAM_DICT)
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

    reward_param = copy.deepcopy(rew_config)

    print("Random Config")
    print(random_param)

    print("Dynamics Config")
    print(dynamic_param)

    print("Sensor Config")
    print(sensor_mode)


    cmd_trj = TrajectoryHandler()

    gait_env = a1sim.make_env(task="none",
                            random_param=random_param, reward_param=reward_param, dynamic_param=dynamic_param,  sensor_mode=sensor_mode, 
                            num_history=sim_config["State History"], render=render, terrain=None, cmd=cmd_trj,
                            normal=1)    

    gait_act_scale = np.array([0.2,0.6,0.6]*4)
    gait_act_offset = np.zeros(12)

    env =TianshouWrapper(gait_env, 
                        action_scale=gait_act_scale, action_offset=gait_act_offset, 
                        max_step = 10e+6, 
                        video_path=None)

    env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_GUI,0)
    env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_SHADOWS,0)

    return PerceptionWrapper(env=env,
                            loco_command=cmd_trj,                            
                            loco_agent=gait_agent,
                            time_step=0.1, time_max=200.0,
                            env_type=env_type, record = record)


if __name__=='__main__':

    env = build_demo_env(render=True)
    env.reset()
    done = False

    while not done:
        
        obs, rew, done, _ = env.step(np.array([0.0, 0.0]))
        img = env.render()