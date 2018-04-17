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
        self.master_losses, self.master_kl = zip(*[self.policy_loss(policies[i], 
            old_policies[i], self.master_obs[i], self.master_acs[i], self.master_atargs[i], 
            self.master_ret[i], clip_param) for i in range(num_master_groups)])

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
        self.sub_losses, self.sub_kl = zip(*[self.policy_loss(sub_policies[i], 
            old_sub_policies[i], self.sub_obs[i], self.sub_acs[i], self.sub_atargs[i], 
            self.sub_ret[i], clip_param) for i in range(num_subpolicies)])

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


    # TODO: check optimizer_scope
    def reset_master_optimizer(self):
        for i in range(self.num_master_groups):
            optimizer_scope = [var for var in 
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    if 'master_adam_%i'%i in var.name] 
            U.get_session().run(tf.initialize_variables(optimizer_scope))
        

    def policy_loss(self, pi, oldpi, ob, ac, atarg, ret, clip_param):
        ratio = tf.exp(pi.pd.logp(ac) - tf.clip_by_value(oldpi.pd.logp(ac), -20, 20)) 
        approx_kl = tf.reduce_mean(tf.square(pi.pd.logp(ac) - oldpi.pd.logp(ac)))
        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        pol_surr = - U.mean(tf.minimum(surr1, surr2))
        vfloss1 = tf.square(pi.vpred - ret)
        vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, 
                clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2))
        total_loss = pol_surr + vf_loss
        return total_loss, approx_kl


    def updateMasterPolicy(self, seg):
        ob, ac, atarg, tdlamret = seg["macro_ob"], seg["macro_ac"], \
                seg["macro_adv"], seg["macro_tdlamret"]
        sample_ob = ob[0][0][0]
        
        def transform_array(array, shape=None):
            array = np.split(array, self.num_master_groups, axis=1)
            if shape != None: 
                array = [elem.reshape(-1, shape) for elem in array]
            else:
                array = [elem.reshape(-1) for elem in array]
            return array

        # ob - T x num_master_groups x num_sub_grps x ob_dims
        # flatten to make train batches
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

        kl_array = []
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

                _, kl = U.get_session().run([self.master_train_steps, self.master_kl], 
                        feed_dict)
                kl_array.append(kl)
        print('KL divergence for the master is %g'%np.mean(kl_array))

        ep_rets = flatten_lists(seg["ep_rets"])
        ep_rets = flatten_lists(ep_rets)
        ep_lens = flatten_lists(seg["ep_lens"])
        ep_lens = flatten_lists(ep_lens)

        return np.mean(ep_rets), np.mean(ep_lens)
        

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
                kl_array = []
                for k in range(self.optim_epochs):
                    for test_batch in test_d.iterate_times(test_batchsize, num_batches):
                        feed_dict = {}
                        feed_dict[self.sub_obs[i]] = test_batch['ob']
                        feed_dict[self.sub_acs[i]] = test_batch['ac']
                        feed_dict[self.sub_atargs[i]] = test_batch['atarg']
                        feed_dict[self.sub_ret[i]] = test_batch['vtarg']

                        _, kl = U.get_session().run([self.sub_train_steps[i], 
                            self.sub_kl[i]], feed_dict)
                        kl_array.append(kl)
                print('KL div for sub %d is %g'%(i, np.mean(kl_array)))
            """
            else:
                # zero grad
                #self.sub_policies[i].ob_rms.noupdate()
                feed_dict = {}
                obs = np.zeros((32, self.ob_space.shape[0]))
                acs, vtargs = self.sub_policies[i].act(False, obs)
                feed_dict[self.sub_obs[i]] = obs 
                feed_dict[self.sub_acs[i]] = acs 
                feed_dict[self.sub_atargs[i]] = np.zeros_like(vtargs)
                feed_dict[self.sub_ret[i]] = vtargs 
                for _ in range(self.optim_epochs):
                    for _ in range(num_batches):
                        U.get_session().run(self.sub_train_steps[i], feed_dict)
            """


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
