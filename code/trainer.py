import gym
import test_envs
import tensorflow as tf
import rollouts
from policy_network import Policy
from subpolicy_network import SubPolicy
from observation_network import Features
from learner import Learner
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines import bench, logger
import rl_algs.common.tf_util as U
import numpy as np
# from tinkerbell import logger
import pickle
import pdb

# TODO: running mean, std for obs in nets
def start(callback, args):
    num_subs = args.num_subs
    macro_duration = args.macro_duration
    num_rollouts = args.num_rollouts
    warmup_time = args.warmup_time
    train_time = args.train_time

    num_master_groups = 1
    # number of batches for the sub-policy optimization
    num_sub_batches = 8
    # number of sub groups in each group
    num_sub_in_grp = 1

    def make_env_vec(seed):
        # common random numbers in sub groups
        def make_env():
            env = gym.make(args.task)
            env.seed(seed)
            if 'Atari' in str(env.__dict__['env']):
                env = wrap_deepmind(env, frame_stack=True)
            #env = bench.Monitor(env, logger.get_dir())
            return env
        return DummyVecEnv([make_env for _ in range(num_sub_in_grp)])

    envs = [make_env_vec(np.random.randint(0, 2**31-1)) for _ in range(num_master_groups)]
    ob_space = envs[0].observation_space
    ac_space = envs[0].action_space

    # observation in.
    master_obs = [U.get_placeholder(name="master_ob_%i"%x, dtype=tf.float32, 
        shape=[None] + list(ob_space.shape)) for x in range(num_master_groups)]
    sub_obs = [U.get_placeholder(name="sub_ob_%i"%x, dtype=tf.float32, 
        shape=[None] + list(ob_space.shape)) for x in range(num_subs)]

    policies = [Policy(name="policy_%i"%x, ob=master_obs[x], ac_space=ac_space, 
        hid_size=32, num_hid_layers=2, num_subpolicies=num_subs) for x in 
        range(num_master_groups)]
    sub_policies = [SubPolicy(name="sub_policy_%i"%x, ob=sub_obs[x], ac_space=ac_space, 
        hid_size=32, num_hid_layers=2) for x in range(num_subs)]

    old_policies = [Policy(name="old_policy_%i"%x, ob=master_obs[x], ac_space=ac_space, 
        hid_size=32, num_hid_layers=2, num_subpolicies=num_subs) for x in 
        range(num_master_groups)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i"%x, ob=sub_obs[x], 
        ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]

    learner = Learner(envs, policies, sub_policies, old_policies, old_sub_policies, 
            clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-4, 
            optim_batchsize=32)
    rollout = rollouts.traj_segment_generator(policies, sub_policies, envs, 
            macro_duration, num_rollouts, num_sub_in_grp, stochastic=True, args=args)


    for x in range(10000):
        callback(x)
        if x == 0:
            [sub_policy.reset() for sub_policy in sub_policies]
            print("synced subpols")

        # Run the inner meta-episode.
        [policy.reset() for policy in policies]
        learner.reset_master_optimizer()

        # TODO: randomizeCorrec() for VecEnv
        #env.env.randomizeCorrect()

        # print("It is iteration %d so i'm changing the goal to %s" % 
        # (x, env.env.realgoal))

        # TODO: is warm-up staggering necessary?
        mini_ep = 0

        totalmeans = []
        while mini_ep < warmup_time+train_time:
            mini_ep += 1
            # rollout
            rolls = rollout.__next__()
            allrolls = []
            allrolls.append(rolls)
            # train theta
            rollouts.add_advantage_macro(rolls, macro_duration, 0.99, 0.98)
            mean, t  = learner.updateMasterPolicy(rolls)
            # train phi
            test_seg = rollouts.prepare_allrolls(allrolls, macro_duration, 0.99, 0.98, 
                    num_subpolicies=num_subs)
            learner.updateSubPolicies(test_seg, num_sub_batches, (mini_ep >= warmup_time))
            # log
            print(("\n%d: rewards %s, episode length %d" % (mini_ep, mean, t)))
            if args.s:
                totalmeans.append(gmean)
                with open('outfile'+str(x)+'.pickle', 'wb') as fp:
                    pickle.dump(totalmeans, fp)
