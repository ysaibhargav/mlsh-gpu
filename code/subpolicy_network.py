import rl_algs.common.tf_util as U
import tensorflow as tf
import numpy as np
import gym
from rl_algs.common.distributions import make_pdtype
from running_mean_std import RunningMeanStd


class SubPolicy(object):
    def _mlp(self, obz, hid_size, num_hid_layers, ac_space, gaussian_fixed_var):
        # value function
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # sub policy
        self.pdtype = pdtype = make_pdtype(ac_space)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "pol%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            self.pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            self.pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(self.pdparam)

    def __init__(self, name, ob, ac_space, network='mlp', gaussian_fixed_var=True):
        hid_size=64
        num_hid_layers=2
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.gaussian_fixed_var = gaussian_fixed_var
        shape = []
        for d in range(1, len(ob.shape)):
            shape.append(ob.shape[d])

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name

            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=shape)
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            if network == 'mlp':
                self._mlp(obz, hid_size, num_hid_layers, ac_space, gaussian_fixed_var)

        # sample actions
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob)#ob[None])
        return ac1, vpred1
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def reset(self):
        with tf.variable_scope(self.scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)
