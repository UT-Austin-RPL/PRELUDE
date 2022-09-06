#! /usr/bin/python3

from typing import Any, Sequence, Tuple
from os import path


## Internal function variables ##

TORQUE = 0
POSITION = 1
VELOCITY = 2


## Visualization modes ##

FRAME_LOCAL = 0
FRAME_GLOBAL = 1


## Simulation modes ##

CONTROL_MPC = 0
CONTROL_QP  = 1

TERRAIN_PLANE = 0
TERRAIN_TRAINING = 1
TERRAIN_VALIDATION = 2
TERRAIN_TEST = 3
TERRAIN_ICE = 4
TERRAIN_SAMURAI = 5
TERRAIN_BLOBBED = 6

OBJECT_EMPTY = 0
OBJECT_BALLS = 1

## Simulation configuartion ##

TIME_STEP = 0.001
REPEAT_INIT = 100
REPEAT_ACTION = 10
NUM_HISTORY = 50


## System configuration ##

PATH_SRC    = path.dirname(path.realpath(__file__))
PATH_ROOT   = path.dirname(PATH_SRC)
PATH_CONFIG = PATH_ROOT+"/config"
PATH_SAVE = PATH_ROOT+"/save"
PATH_DATA   = PATH_ROOT+"/data"
PATH_PLOT   = PATH_ROOT+"/plot"

SUBPATH_CONFIG = {  "reward":   "reward.yaml",
                    "ppo":      "ppo.yaml",
                    "experiment": "experiment.yaml"}


SUBPATH_TERRAIN = { TERRAIN_SAMURAI:   "terrains/samurai/samurai.urdf",
                    TERRAIN_BLOBBED:   "terrains/blobbed_terrain/terrain.urdf"}


PATH_TERRAIN_TRAINING=PATH_DATA+'/dataset'
PATH_TERRAIN_VALIDATION=PATH_DATA+'/dataset'


