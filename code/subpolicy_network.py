import rl_algs.common.tf_util as U
import tensorflow as tf
import numpy as np
import gym
from rl_algs.common.distributions import make_pdtype
from running_mean_std import RunningMeanStd
from baselines.a2c.utils import conv, fc, conv_to_fc, \
        batch_to_seq, seq_to_batch, lstm, lnlstm
#from baselines.ppo2.policies import nature_cnn
import pdb

def feature_net(unscaled_images):
    scaled_images = tf.cast(unscaled_images, tf.float32)# / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=4, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=2, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=16, init_scale=np.sqrt(2)))

class SubPolicy(object):
    def _mlp(self, obs, hid_size, num_hid_layers, ac_space, gaussian_fixed_var):
        # value function
        last_out = obs
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), 
                weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", 
                weight_init=U.normc_initializer(1.0))[:,0]

        # sub policy
        self.pdtype = pdtype = make_pdtype(ac_space)
        last_out = obs
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "pol%i"%(i+1), 
                weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", 
                    U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], 
                    initializer=tf.zeros_initializer())
            self.pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            self.pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", 
                    U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(self.pdparam)

    def _lstm(self, obs, states, masks, nlstm, ac_space, nbatch, nsteps, reuse=False):
        # obs: nbatch * ob_shape 
        # states: num_env * (2xnlstm) 
        # masks: nbatch
        # TODO: fix dimensions
        num_env = nbatch // nsteps
        nh, nw, nc = obs.shape[1:]
        ob_shape = [nsteps, nh, nw, nc]
        nact = ac_space.n
        with tf.variable_scope('lstm', reuse=reuse):
            h = feature_net(obs)
            xs = batch_to_seq(h, num_env, nsteps)
            ms = batch_to_seq(masks, num_env, nsteps)
            h5, snew = lstm(xs, ms, states, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.vpred = vf[:, 0]

        self.pdtype = pdtype = make_pdtype(ac_space)
        self.pd = pdtype.pdfromflat(pi)

        self.snew = snew

    def __init__(self, name, ob, ac_space, network='mlp', gaussian_fixed_var=True, 
            nsteps=None, nbatch=None, nlstm=256, states=None, masks=None, reuse=False):
        self.network = network

        shape = []
        for d in range(1, len(ob.shape)):
            shape.append(ob.shape[d])

        with tf.variable_scope(name, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=shape)
            obs = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            if network == 'mlp':
                hid_size=64
                num_hid_layers=2
                self.hid_size = hid_size
                self.num_hid_layers = num_hid_layers
                self.gaussian_fixed_var = gaussian_fixed_var
                self._mlp(obs, hid_size, num_hid_layers, ac_space, gaussian_fixed_var)
            elif network == 'lstm':
                assert nsteps is not None and nbatch is not None
                assert states is not None and masks is not None
                assert isinstance(nsteps, int) and isinstance(nbatch, int)
                assert nsteps > 0 and nbatch > 0
                self._lstm(obs, states, masks, nlstm, ac_space, nbatch, nsteps)


        # sample actions
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        if network == 'mlp':
            self._act = U.function([stochastic, ob], [ac, self.vpred])
        elif network == 'lstm':
            self._act = U.function([stochastic, ob, states, masks], 
                    [ac, self.vpred, self.snew])

    def act(self, stochastic, ob, states=None, masks=None):
        if self.network == 'mlp':
            ac1, vpred1 = self._act(stochastic, ob)
            return ac1, vpred1
        elif self.network == 'lstm':
            ac1, vpred1, snew = self._act(stochastic, ob, states, masks)
            return ac1, vpred1, snew
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def reset(self):
        with tf.variable_scope(self.scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)
