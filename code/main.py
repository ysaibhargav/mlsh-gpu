import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument('savename', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--num_subs', type=int)
parser.add_argument('--macro_duration', type=int)
parser.add_argument('--num_rollouts', type=int)
parser.add_argument('--warmup_time', type=int)
parser.add_argument('--num_master_grps', type=int)
parser.add_argument('--num_sub_batches', type=int)
parser.add_argument('--num_sub_in_grp', type=int)
parser.add_argument('--vfcoeff', type=float)
parser.add_argument('--entcoeff', type=float)
parser.add_argument('--master_lr', type=float)
parser.add_argument('--sub_lr', type=float)
parser.add_argument('--train_time', type=int)
parser.add_argument('--force_subpolicy', type=int)
parser.add_argument('--replay', type=str)
parser.add_argument('--continue_iter', type=str)
args = parser.parse_args()

from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import os
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
from baselines.logger import Logger, CSVOutputFormat, HumanOutputFormat

# python3 main.py --task=CartPole-v0 --num_subs=1 --macro_duration=10 --num_rollouts=1000 --warmup_time=2 --train_time=190 --replay=n --num_master_grp=1 --num_sub_batches=8 --num_sub_in_grp=1 --vfcoeff=2. --entcoeff=0 cp

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
LOGDIR = osp.join("savedir", args.savename, 'logs')
CKPTDIR = osp.join("savedir", args.savename, 'checkpoints')

def callback(it):
    if it % 5 == 0 and it > 3 and not replay:
        fname = osp.join(CKPTDIR, '%.5i'%it)
        U.save_state(fname)
    if args.continue_iter is not None and int(args.continue_iter)+1 == it:
        fname = osp.join(CKPTDIR, str(args.continue_iter))
        U.load_state(fname)

def train():
    num_timesteps=1e9

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    trainer.start(callback, args=args)

def main():
    if osp.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    os.makedirs(LOGDIR)
    if not osp.exists(CKPTDIR):
        os.makedirs(CKPTDIR)
    Logger.DEFAULT = Logger.CURRENT = Logger(dir=None, 
            output_formats=[HumanOutputFormat(sys.stdout), 
                CSVOutputFormat(osp.join(LOGDIR, 'log.csv'))])
    train()

if __name__ == '__main__':
    main()
