import gymnasium as gym
import ale_py

def make_env():
    gym.register_envs(ale_py)
    return gym.make("ALE/MsPacman-v5", full_action_space=False)
