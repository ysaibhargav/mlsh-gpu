import gym
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import bench, logger

def make_env():
    env = gym.make('CartPole-v0')
    env = bench.Monitor(env, logger.get_dir())
    return env
env = DummyVecEnv([make_env for _ in range(2)])
