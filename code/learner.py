import numpy as np
import tensorflow as tf
from rl_algs.common import explained_variance, fmt_row, zipsame
from rl_algs import logger
import rl_algs.common.tf_util as U
import time
from rl_algs.common.mpi_adam import MpiAdam
from mpi4py import MPI
from collections import deque
from dataset import Dataset
import pdb

class Learner:
    def __init__(self, envs, policies, sub_policies, old_policies, old_sub_policies, 
            clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-4, 
            optim_batchsize=64):
        self.policies = policies
        self.sub_policies = sub_policies
        self.old_policies = old_policies
        self.old_sub_policies = old_sub_policies
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.num_master_groups = num_master_groups = len(policies)
        self.num_subpolicies = num_subpolicies = len(sub_policies)
        self.ob_space = envs[0].observation_space
        self.ac_space = envs[0].action_space

        self.master_obs = [U.get_placeholder(name="master_ob_%i"%x, dtype=tf.float32,
            shape=[None, self.ob_space.shape[0]]) for x in range(num_master_groups)]
        self.master_acs = [policies[0].pdtype.sample_placeholder([None]) 
                for _ in range(num_master_groups)]
        self.master_atargs = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_master_groups)]
        self.master_ret = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_master_groups)]
        self.master_losses = [self.policy_loss(policies[i], old_policies[i], 
            self.master_obs[i], self.master_acs[i], self.master_atargs[i], 
            self.master_ret[i], clip_param) for i in range(num_master_groups)]

        master_trainers = [tf.train.AdamOptimizer(learning_rate=0.01, 
            name='master_adam_%i'%_) for _ in range(num_master_groups)]
        master_params = [policies[i].get_trainable_variables() 
                for i in range(num_master_groups)] 
        master_grads = [tf.gradients(self.master_losses[i], master_params[i])
                for i in range(num_master_groups)]
        master_grads = [list(zip(g, p)) for g, p in zip(master_grads, master_params)]
        # TODO: gradient clipping
        self.assign_old_eq_new = [U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(old_policies[i].get_variables(), 
                policies[i].get_variables())]) for i in range(num_master_groups)]
        self.master_train_steps = [master_trainers[i].apply_gradients(master_grads[i])
                for i in range(num_master_groups)]
       

        self.sub_obs = [U.get_placeholder(name="sub_ob_%i"%x, dtype=tf.float32,
            shape=[None, self.ob_space.shape[0]]) for x in range(num_subpolicies)]
        self.sub_acs = [sub_policies[0].pdtype.sample_placeholder([None]) 
                for _ in range(num_subpolicies)]
        self.sub_atargs = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_subpolicies)]
        self.sub_ret = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_subpolicies)]
        self.sub_losses = [self.policy_loss(sub_policies[i], old_sub_policies[i], 
            self.sub_obs[i], self.sub_acs[i], self.sub_atargs[i], self.sub_ret[i], 
            clip_param) for i in range(num_subpolicies)]

        sub_trainers = [tf.train.AdamOptimizer(learning_rate=optim_stepsize)
                for _ in range(num_subpolicies)]
        sub_params = [sub_policies[i].get_trainable_variables() 
                for i in range(num_subpolicies)] 
        sub_grads = [tf.gradients(self.sub_losses[i], sub_params[i])
                for i in range(num_subpolicies)]
        sub_grads = [list(zip(g, p)) for g, p in zip(sub_grads, sub_params)]
        # TODO: gradient clipping
        self.subs_assign_old_eq_new = [U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(old_sub_policies[i].get_variables(), 
                sub_policies[i].get_variables())]) for i in range(num_subpolicies)]
        self.sub_train_steps = [sub_trainers[i].apply_gradients(sub_grads[i])
                for i in range(num_subpolicies)]

        U.initialize()

        # TODO: dummy gradient update for sub-policies
        """
        self.assign_subs = []
        self.change_subs = []
        self.adams = []
        self.losses = []
        self.sp_ac = sub_policies[0].pdtype.sample_placeholder([None])
        for i in range(self.num_subpolicies):
            varlist = sub_policies[i].get_trainable_variables()
            self.adams.append(MpiAdam(varlist))
            # loss for test
            loss = self.policy_loss(sub_policies[i], old_sub_policies[i], ob, self.sp_ac, atarg, ret, clip_param)
            self.losses.append(U.function([ob, self.sp_ac, atarg, ret], U.flatgrad(loss, varlist)))

            self.assign_subs.append(U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(old_sub_policies[i].get_variables(), sub_policies[i].get_variables())]))
            self.zerograd = U.function([], self.nograd(varlist))


        self.master_adam.sync()
        for i in range(self.num_subpolicies):
            self.adams[i].sync()

    def nograd(self, var_list):
        return tf.concat(axis=0, values=[
            tf.reshape(tf.zeros_like(v), [U.numel(v)])
            for v in var_list
        ])
        """

    # TODO: check optimizer_scope
    def reset_master_optimizer(self):
        for i in range(self.num_master_groups):
            optimizer_scope = [var for var in 
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    if 'master_adam_%i'%i in var.name] 
            U.get_session().run(tf.initialize_variables(optimizer_scope))
        

    def policy_loss(self, pi, oldpi, ob, ac, atarg, ret, clip_param):
        ratio = tf.exp(pi.pd.logp(ac) - tf.clip_by_value(oldpi.pd.logp(ac), -20, 20)) 
        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        pol_surr = - U.mean(tf.minimum(surr1, surr2))
        vfloss1 = tf.square(pi.vpred - ret)
        vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, 
                clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2))
        total_loss = pol_surr + vf_loss
        return total_loss


    def updateMasterPolicy(self, seg):
        ob, ac, atarg, tdlamret = seg["macro_ob"], seg["macro_ac"], \
                seg["macro_adv"], seg["macro_tdlamret"]
        sample_ob = ob[0][0][0]
        
        def transform_array(array, shape=None):
            array = np.split(array, self.num_master_groups, axis=1)
            if shape != None: 
                # TODO: check if this logic is correct
                array = [elem.reshape(-1, shape) for elem in array]
            else:
                array = [elem.reshape(-1) for elem in array]
            return array

        ob = transform_array(ob, int(sample_ob.shape[0])) 
        ac = transform_array(ac)
        atarg = transform_array(atarg)
        tdlamret = transform_array(tdlamret) 

        atarg = np.array(atarg, dtype='float32')
        mean = atarg.mean()
        std = atarg.std()
        atarg = (atarg - mean) / max(std, 0.000001)

        d = [Dataset(dict(ob=ob[i], ac=ac[i], atarg=atarg[i], vtarg=tdlamret[i]), 
            shuffle=True) for i in range(self.num_master_groups)]
        optim_batchsize = min(self.optim_batchsize, ob[0].shape[0])
        num_updates = ob[0].shape[0] // optim_batchsize

        [self.policies[i].ob_rms.update(ob[i]) for i in range(self.num_master_groups)]
        [f() for f in self.assign_old_eq_new]

        for _ in range(self.optim_epochs):
            for __ in range(num_updates):
                batches = [next(d[i].iterate_once(optim_batchsize))
                        for i in range(self.num_master_groups)]
                feed_dict = {}
                for i in range(self.num_master_groups):
                    feed_dict[self.master_obs[i]] = batches[i]['ob']
                    feed_dict[self.master_acs[i]] = batches[i]['ac']
                    feed_dict[self.master_atargs[i]] = batches[i]['atarg']
                    feed_dict[self.master_ret[i]] = batches[i]['vtarg']

                U.get_session().run(self.master_train_steps, feed_dict)

        """
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        logger.record_tabular("EpRewMean", np.mean(rews))

        return np.mean(rews), np.mean(seg["ep_rets"])
        """
        return np.mean(seg['ep_rets'].reshape(-1))
        

    def updateSubPolicies(self, test_segs, num_batches, optimize=True):
        for i in range(self.num_subpolicies):
            is_optimizing = True
            test_seg = test_segs[i]
            ob, ac, atarg, tdlamret = test_seg["ob"], test_seg["ac"], test_seg["adv"], \
                    test_seg["tdlamret"]
            if np.shape(ob)[0] < 1:
                is_optimizing = False
            else:
                atarg = np.array(atarg, dtype='float32')
                atarg = (atarg - atarg.mean()) / max(atarg.std(), 0.000001)
            test_d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
            test_batchsize = int(ob.shape[0] / num_batches)

            self.subs_assign_old_eq_new[i]()

            if self.optim_batchsize > 0 and is_optimizing and optimize:
                self.sub_policies[i].ob_rms.update(ob)
                for k in range(self.optim_epochs):
                    for test_batch in test_d.iterate_times(test_batchsize, num_batches):
                        feed_dict = {}
                        feed_dict[self.sub_obs[i]] = test_batch['ob']
                        feed_dict[self.sub_acs[i]] = test_batch['ac']
                        feed_dict[self.sub_atargs[i]] = test_batch['atarg']
                        feed_dict[self.sub_ret[i]] = test_batch['vtarg']

                    U.get_session().run(self.sub_train_steps, feed_dict)
            """
            else:
                self.sub_policies[i].ob_rms.noupdate()
                blank = self.zerograd()
                for _ in range(self.optim_epochs):
                    for _ in range(num_batches):
                        self.adams[i].update(blank, self.optim_stepsize, 0)
            """

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
