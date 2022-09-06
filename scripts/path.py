import os
import yaml

PATH_SICRIPT    = os.path.dirname(os.path.realpath(__file__))
PATH_ROOT   = os.path.dirname(PATH_SICRIPT)
SUBPATH = yaml.load(open(os.path.join(PATH_ROOT, 'path.yaml')), Loader=yaml.FullLoader)

PATH_DATA   = os.path.join(PATH_ROOT, SUBPATH['Data'])
PATH_CONFIG = os.path.join(PATH_ROOT, SUBPATH['Configuration'])

PATH_CHECKPOINT_RL = os.path.join(PATH_ROOT, SUBPATH['RL Checkpoint'])
PATH_CHECKPOINT_BC = os.path.join(PATH_ROOT, SUBPATH['BC Checkpoint'])

PATH_DATASETSET_SIM   = os.path.join(PATH_ROOT, SUBPATH['Dataset Sim'])
PATH_DATASETSET_REAL   = os.path.join(PATH_ROOT, SUBPATH['Dataset Real'])

PATH_RAW_SIM  = os.path.join(PATH_ROOT, SUBPATH['Raw Data Sim'])
PATH_RAW_REAL = os.path.join(PATH_ROOT, SUBPATH['Raw Data Real'])
