
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .g1.g1_config import G1RoughCfg, G1RoughCfgPPO



import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO() )


