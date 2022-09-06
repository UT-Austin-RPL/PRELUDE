"""Set up gym interface for locomotion environments."""
from a1sim.envs import *

def make_env(**kwargs):
    return A1GymEnv(**kwargs)