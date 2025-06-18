
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .g1.g1_config import G1RoughCfg, G1RoughCfgPPO, G1RoughCfgPPOEMLP, G1RoughCfgPPOHalf_symmetry, G1RoughCfgPPOEMLPVae



import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO() )

task_registry.register( "g1-emlp", LeggedRobot, G1RoughCfg(), G1RoughCfgPPOEMLP() )

task_registry.register( "g1-halfsym", LeggedRobot, G1RoughCfg(), G1RoughCfgPPOHalf_symmetry() )

task_registry.register( "g1-emlp_vae", LeggedRobot, G1RoughCfg(), G1RoughCfgPPOEMLPVae() )


