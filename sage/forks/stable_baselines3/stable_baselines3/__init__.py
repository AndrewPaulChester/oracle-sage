import os

from sage.forks.stable_baselines3.stable_baselines3.a2c import A2C
from sage.forks.stable_baselines3.stable_baselines3.ddpg import DDPG
from sage.forks.stable_baselines3.stable_baselines3.dqn import DQN
from sage.forks.stable_baselines3.stable_baselines3.her import HER
from sage.forks.stable_baselines3.stable_baselines3.ppo import PPO
from sage.forks.stable_baselines3.stable_baselines3.sac import SAC
from sage.forks.stable_baselines3.stable_baselines3.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
