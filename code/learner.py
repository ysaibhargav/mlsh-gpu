import numpy as np
import tensorflow as tf
from rl_algs.common import explained_variance, fmt_row, zipsame
from baselines import logger
import rl_algs.common.tf_util as U
import time
from rl_algs.common.mpi_adam import MpiAdam
from mpi4py import MPI
from collections import deque
from dataset import Dataset
import pdb
import sys

class Learner:
    def __init__(self, envs, policies, sub_policies, old_policies, old_sub_policies, 
            clip_param=0.2, vfcoeff=1., entcoeff=0, divcoeff=0., optim_epochs=10, 
            master_lr=1e-3, sub_lr=3e-4, optim_batchsize=64, envsperbatch=None, 
            num_rollouts=None, nlstm=256, recurrent=False):
        self.policies = policies
        self.sub_policies = sub_policies
        self.old_policies = old_policies
        self.old_sub_policies = old_sub_policies
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_batchsize = optim_batchsize
        self.num_master_groups = num_master_groups = len(policies)
        self.num_subpolicies = num_subpolicies = len(sub_policies)
        self.ob_space = envs[0].observation_space
        self.ac_space = envs[0].action_space
        self.nbatch = nbatch = num_rollouts * envsperbatch
        self.envsperbatch = envsperbatch

        self.master_obs = [U.get_placeholder(name="master_ob_%i"%x, dtype=tf.float32,
            shape=[None] + list(self.ob_space.shape)) for x in range(num_master_groups)]
        self.master_acs = [policies[0].pdtype.sample_placeholder([None]) 
                for _ in range(num_master_groups)]
        self.master_atargs = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_master_groups)]
        self.master_ret = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_master_groups)]
        retvals = zip(*[self.policy_loss(policies[i], 
            old_policies[i], self.master_obs[i], self.master_acs[i], self.master_atargs[i], 
            self.master_ret[i], clip_param, mask=tf.constant(1.), vfcoeff=vfcoeff, 
            entcoeff=entcoeff) for i in range(num_master_groups)])
        self.master_losses, self.master_kl, self.master_pol_surr, self.master_vf_loss, \
                self.master_entropy, self.master_values, _ = retvals 

        master_trainers = [tf.train.AdamOptimizer(learning_rate=master_lr, 
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
       

        if not recurrent:
            self.sub_obs = [U.get_placeholder(name="sub_ob_%i"%x, dtype=tf.float32,
                shape=[None] + list(self.ob_space.shape)) for x in range(num_subpolicies)]
        self.sub_acs = [sub_policies[0].pdtype.sample_placeholder([None]) 
                for _ in range(num_subpolicies)]
        self.sub_atargs = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_subpolicies)]
        self.sub_ret = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_subpolicies)]
        self.logpacs = [tf.placeholder(dtype=tf.float32, shape=[num_subpolicies, None])
                for _ in range(num_subpolicies)]
        self.loss_masks = [tf.placeholder(dtype=tf.float32, shape=[None])
                for _ in range(num_subpolicies)]
        if recurrent:
            self.sub_obs = [U.get_placeholder(name="sub_ob_%i"%x, dtype=tf.float32,
                shape=[nbatch] + list(self.ob_space.shape)) for x in range(num_subpolicies)]
            self.sub_masks = [U.get_placeholder(name="masks_%i"%_, dtype=tf.float32, 
                shape=[nbatch]) for _ in range(num_subpolicies)]
            self.sub_states = [U.get_placeholder(name="states_%i"%_, dtype=tf.float32, 
                shape=[envsperbatch, 2*nlstm]) for _ in range(num_subpolicies)]
        sub_retvals = zip(*[self.policy_loss(sub_policies[i], 
            old_sub_policies[i], self.sub_obs[i], self.sub_acs[i], self.sub_atargs[i], 
            self.sub_ret[i], clip_param, mask=self.loss_masks[i], vfcoeff=vfcoeff, 
            entcoeff=entcoeff, divcoeff=divcoeff, logpacs=None)#self.logpacs[i]) 
            for i in range(num_subpolicies)])
        self.sub_losses, self.sub_kl, self.sub_pol_surr, self.sub_vf_loss, \
                self.sub_entropy, self.sub_values, self.div_loss = sub_retvals 

        sub_trainers = [tf.train.AdamOptimizer(learning_rate=sub_lr)
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
        
    # TODO: cross check with baselines PPO
    def policy_loss(self, pi, oldpi, ob, ac, atarg, ret, clip_param, mask=1, vfcoeff=1., 
            entcoeff=0, divcoeff=0., logpacs=None):
        LOGP_MAX = 20
        KL_MAX = 5
        entropy = tf.reduce_mean(pi.pd.entropy())
        ratio = tf.exp(pi.pd.logp(ac) - U.clip(oldpi.pd.logp(ac), -LOGP_MAX, LOGP_MAX)) 
        approx_kl = tf.reduce_mean(tf.square(pi.pd.logp(ac) - oldpi.pd.logp(ac)))
        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        pol_surr = -U.mean(mask*tf.minimum(surr1, surr2))
        vfloss1 = tf.square(pi.vpred - ret)
        vpredclipped = oldpi.vpred + U.clip(pi.vpred - oldpi.vpred, -clip_param, 
                clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        vf_loss = U.mean(mask*tf.maximum(vfloss1, vfloss2))
        mask_scalar = tf.reduce_mean(mask)
        total_loss = pol_surr + vfcoeff*vf_loss - mask_scalar*entcoeff*entropy
        div_loss = None
        if logpacs is not None:
            div_loss = tf.reduce_sum(U.clip(tf.reduce_mean(tf.square(tf.exp(pi.pd.logp(ac))-
                U.clip(tf.exp(logpacs), -LOGP_MAX, LOGP_MAX))*mask, axis=1), 0, KL_MAX))
            total_loss -= divcoeff*div_loss
        return total_loss, approx_kl, pol_surr, vf_loss, entropy, pi.vpred, div_loss


    # TODO: explained variance
    def updateMasterPolicy(self, seg, optimize=False):
        ep_rets = flatten_lists(seg["ep_rets"])
        ep_rets = flatten_lists(ep_rets)
        ep_lens = flatten_lists(seg["ep_lens"])
        ep_lens = flatten_lists(ep_lens)

        logger.logkv('Mean episode return', np.mean(ep_rets))
        logger.logkv('Mean episode length', np.mean(ep_lens))
        logger.dumpkvs()

        if optimize:
            ob, ac, atarg, tdlamret = seg["macro_ob"], seg["macro_ac"], \
                    seg["macro_adv"], seg["macro_tdlamret"]
            sample_ob = ob[0][0][0]
            
            def transform_array(array, shape=None):
                array = np.split(array, self.num_master_groups, axis=1)
                if shape != None: 
                    shape = [-1] + shape
                    array = [elem.reshape(*shape) for elem in array]
                else:
                    array = [elem.reshape(-1) for elem in array]
                return array

            # ob - T x num_master_groups x num_sub_grps x ob_dims
            # flatten to make train batches
            ob = transform_array(ob, list(sample_ob.shape)) 
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

            kl_array, pol_surr_array, vf_loss_array, entropy_array, values_array = [[] for _ in 
                    range(5)]
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

                    _, kl, pol_surr, vf_loss, entropy, values = U.get_session().run(
                            [self.master_train_steps, 
                        self.master_kl, self.master_pol_surr, self.master_vf_loss, 
                        self.master_entropy, self.master_values], feed_dict)
                    kl_array.append(kl)
                    pol_surr_array.append(pol_surr)
                    vf_loss_array.append(vf_loss)
                    entropy_array.append(entropy)

            logger.logkv('(M) KL', np.mean(kl_array))
            logger.logkv('(M) policy loss', np.mean(pol_surr_array))
            logger.logkv('(M) value loss', np.mean(vf_loss_array))
            logger.logkv('(M) entropy loss', np.mean(entropy_array))
            logger.dumpkvs()

    def updateSubPoliciesRecurrent(self, test_segs, num_batches, horizon, num_env, 
            optimize=True):
        envinds = np.arange(num_env)
        flatinds = np.arange(num_env*horizon).reshape(num_env, horizon)
        envsperbatch = max(1, num_env // num_batches)
        num_batches = num_env // envsperbatch
        for i in range(len(self.sub_policies)):
            test_seg = test_segs[i]
            ob, ac, atarg, tdlamret, new, mask = test_seg["ob"], test_seg["ac"], \
                    test_seg["adv"], test_seg["tdlamret"], test_seg["new"], test_seg["mask"]
            mask_idx = np.array(mask, bool)
            is_optimizing = np.sum(mask) > 0

            self.subs_assign_old_eq_new[i]()

            if self.optim_batchsize > 0 and is_optimizing and optimize:
                self.sub_policies[i].ob_rms.update(ob[mask_idx])
                atarg = np.array(atarg, dtype='float32')
                atarg = (atarg - atarg[mask_idx].mean()) / max(atarg[mask_idx].std(), 
                        0.000001)
                test_seg["adv"] = atarg

        def make_feed_dict():
            feed_dict = [{} for _ in range(num_batches)]

            for i in range(self.num_subpolicies):
                test_seg = test_segs[i]
                np.random.shuffle(envinds)
                batch_num = 0
                for start in range(0, num_env, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()

                    feed_dict[batch_num][self.sub_obs[i]] = test_seg["ob"][mbflatinds] 
                    feed_dict[batch_num][self.sub_acs[i]] = test_seg["ac"][mbflatinds] 
                    feed_dict[batch_num][self.sub_atargs[i]] = test_seg["adv"][mbflatinds] 
                    feed_dict[batch_num][self.sub_ret[i]] = test_seg["tdlamret"][mbflatinds] 
                    feed_dict[batch_num][self.sub_masks[i]] = test_seg["new"][mbflatinds] 
                    feed_dict[batch_num][self.loss_masks[i]] = test_seg["mask"][mbflatinds] 
                    feed_dict[batch_num][self.sub_states[i]] = test_seg["state"][mbenvinds] 
                    batch_num += 1

            return feed_dict


        if optimize:
            kl_array, pol_surr_array, vf_loss_array, entropy_array = [[[] 
                for _ in range(self.num_subpolicies)] for _ in range(4)]
            for _ in range(self.optim_epochs):
                feed_dict = make_feed_dict()
                for _dict in feed_dict:

                 _, sub_kl, sub_pol_surr, sub_vf_loss, sub_entropy = \
                        U.get_session().run([self.sub_train_steps, self.sub_kl, 
                            self.sub_pol_surr, self.sub_vf_loss, self.sub_entropy], _dict)

            valid_idx = range(self.num_subpolicies)
            ix_append_(kl_array, sub_kl, valid_idx)
            ix_append_(pol_surr_array, sub_pol_surr, valid_idx)
            ix_append_(vf_loss_array, sub_vf_loss, valid_idx)
            ix_append_(entropy_array, sub_entropy, valid_idx)
                    
            for i in range(self.num_subpolicies):
                logger.logkv('(S%d) KL'%i, np.mean(kl_array[i]))
                logger.logkv('(S%d) policy loss'%i, np.mean(pol_surr_array[i]))
                logger.logkv('(S%d) value loss'%i, np.mean(vf_loss_array[i]))
                logger.logkv('(S%d) entropy loss'%i, np.mean(entropy_array[i]))
                logger.dumpkvs()


    def updateSubPoliciesNonRecurrent(self, test_segs, num_batches, optimize=True):
        optimizable = []
        test_ds = []
        batchsizes = []
        for i in range(self.num_subpolicies):
            is_optimizing = True
            test_seg = test_segs[i]
            ob, ac, atarg, tdlamret, mask = test_seg["ob"], test_seg["ac"], test_seg["adv"], \
                    test_seg["tdlamret"], test_seg["mask"]
            # logpacs for diversity loss
            logpacs = [U.get_session().run(pi.pd.logp(test_seg['ac']), 
                {self.sub_obs[j]: test_seg['ob']}) for j, pi in enumerate(self.sub_policies)]
            logpacs = np.array(logpacs).transpose()

            # don't optimize if insufficient data
            if np.shape(ob)[0] < 1:
                is_optimizing = False
            else:
                atarg = np.array(atarg, dtype='float32')
                atarg = (atarg - atarg.mean()) / max(atarg.std(), 0.000001)
            test_d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, 
                logpacs=logpacs, mask=mask), shuffle=True)
            test_batchsize = int(ob.shape[0] / num_batches)

            optimizable.append(is_optimizing)
            test_ds.append(test_d)
            batchsizes.append(test_batchsize)

            self.subs_assign_old_eq_new[i]()

            if self.optim_batchsize > 0 and is_optimizing and optimize:
                self.sub_policies[i].ob_rms.update(ob)

        if optimize:
            def make_feed_dict():
                # one dict per batch
                # multiple subpolicy data in one dict
                feed_dict = [{} for _ in range(num_batches)]

                for i in range(self.num_subpolicies):
                    if self.optim_batchsize > 0 and optimizable[i]:
                        batch_num = 0
                        for test_batch in test_ds[i].iterate_times(batchsizes[i], 
                                num_batches):
                            feed_dict[batch_num][i] = True
                            feed_dict[batch_num][self.sub_obs[i]] = test_batch['ob']
                            feed_dict[batch_num][self.sub_acs[i]] = test_batch['ac']
                            feed_dict[batch_num][self.sub_atargs[i]] = test_batch['atarg']
                            feed_dict[batch_num][self.sub_ret[i]] = test_batch['vtarg']
                            feed_dict[batch_num][self.loss_masks[i]] = test_batch['mask']
                            feed_dict[batch_num][self.logpacs[i]] = \
                                    test_batch['logpacs'].transpose()
                            batch_num += 1

                return feed_dict

            kl_array, pol_surr_array, vf_loss_array, entropy_array, div_array = [[[] for _ 
                in range(self.num_subpolicies)] for _ in range(5)]
            for _ in range(self.optim_epochs):
                feed_dict = make_feed_dict()
                for _dict in feed_dict:
                    # get indicies of subpolicies whose data exists
                    valid_idx = [i for i in range(self.num_subpolicies) if i in _dict] 
                    train_steps = ix_(self.sub_train_steps, valid_idx)  
                    kl = ix_(self.sub_kl, valid_idx)  
                    pol_surr = ix_(self.sub_pol_surr, valid_idx)  
                    vf_loss = ix_(self.sub_vf_loss, valid_idx)  
                    entropy = ix_(self.sub_entropy, valid_idx)  
                    div_loss = ix_(self.div_loss, valid_idx)
                    
                    keys = list(_dict.keys())
                    for key in keys:
                        if isinstance(key, int):
                            del _dict[key]

                    """
                    _, sub_kl, sub_pol_surr, sub_vf_loss, sub_entropy, sub_div_loss = \
                            U.get_session().run([train_steps, kl, pol_surr, vf_loss, 
                                entropy, div_loss], _dict)
                    """
                    _, sub_kl, sub_pol_surr, sub_vf_loss, sub_entropy= \
                            U.get_session().run([train_steps, kl, pol_surr, vf_loss, 
                                entropy], _dict)

                    ix_append_(kl_array, sub_kl, valid_idx)
                    ix_append_(pol_surr_array, sub_pol_surr, valid_idx)
                    ix_append_(vf_loss_array, sub_vf_loss, valid_idx)
                    ix_append_(entropy_array, sub_entropy, valid_idx)
                    #ix_append_(div_array, sub_div_loss, valid_idx)

            for i in range(self.num_subpolicies):
                logger.logkv('(S%d) KL'%i, np.mean(kl_array[i]))
                logger.logkv('(S%d) policy loss'%i, np.mean(pol_surr_array[i]))
                logger.logkv('(S%d) value loss'%i, np.mean(vf_loss_array[i]))
                logger.logkv('(S%d) entropy loss'%i, np.mean(entropy_array[i]))
                #logger.logkv('(S%d) diversity loss'%i, np.mean(div_array[i]))
                logger.dumpkvs()


    def updateSubPolicies(self, test_segs, num_batches, horizon, num_env, 
            optimize=True, recurrent=False):
        if recurrent:
            self.updateSubPoliciesRecurrent(test_segs, num_batches, horizon, num_env, 
                    optimize=optimize)
        else:
            self.updateSubPoliciesNonRecurrent(test_segs, num_batches, optimize=optimize)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def ix_(a, b):
    return [a[_b] for _b in b]


def ix_append_(a, b, c):
    for _c, _b in zip(c, b):
        a[_c].append(_b)
