import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument('savename', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--num_subs', type=int)
parser.add_argument('--macro_duration', type=int)
parser.add_argument('--num_rollouts', type=int)
parser.add_argument('--warmup_time', type=int)
parser.add_argument('--train_time', type=int)
parser.add_argument('--force_subpolicy', type=int)
parser.add_argument('--replay', type=str)
parser.add_argument('-s', action='store_true')
parser.add_argument('--continue_iter', type=str)
args = parser.parse_args()

from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import misc_util
import sys
import shutil
import subprocess
import trainer 
import multiprocessing

"""
python3 main.py --task KeyDoor-v0 --num_subs 2 --macro_duration 1000 --num_rollouts 2000 --warmup_time 20 --train_time 30 --replay False KeyDoor 
"""

# TODO: logging
# TODO: Pacman integration
# TODO: num_rollouts? 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

replay = str2bool(args.replay)
args.replay = str2bool(args.replay)

RELPATH = osp.join(args.savename)
LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') 
        else '/tmp', RELPATH)

def callback(it):
    if it % 5 == 0 and it > 3 and not replay:
        fname = osp.join("savedir", args.savename, 'checkpoints', '%.5i'%it)
        U.save_state(fname)
    if it == 0 and args.continue_iter is not None:
        fname = osp.join("savedir", args.savename, "checkpoints", 
                str(args.continue_iter))
        U.load_state(fname)

def train():
    num_timesteps=1e9

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    # TODO: parallelism?
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    trainer.start(callback, args=args)

def main():
    if osp.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    train()

if __name__ == '__main__':
    main()
