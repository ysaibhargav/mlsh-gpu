import numpy as np
import math
import time
import pdb

def traj_segment_generator(policies, sub_policies, envs, macrolen, horizon, 
        num_sub_in_grp, stochastic, args):
    recurrent = args.subpol_network == 'lstm'

    EPS = 0.1

    num_master_groups = len(policies)
    replay = args.replay
    t = 0
    # ac - num_master_groups * num_sub_in_grp * num_policies
    ac = [[envs[0].action_space.sample() for _ in range(num_sub_in_grp)] for env in envs]
    ac = np.array(ac, dtype='int32')
    vpred = np.zeros([num_master_groups, num_sub_in_grp], dtype='float32')
    vpred2 = np.zeros([num_master_groups, num_sub_in_grp], dtype='float32')
    new = [[False for _ in range(num_sub_in_grp)] for _ in range(num_master_groups)] 
    rew = [[0.0 for _ in range(num_sub_in_grp)] for _ in range(num_master_groups)] 
    # ob - num_master_groups * num_sub_in_grp * state_dims
    ob = [env.reset() for env in envs]
    macro_horizon = math.ceil(horizon/macrolen)

    cur_ep_ret = np.zeros([num_master_groups, num_sub_in_grp], dtype='float32')
    cur_ep_len = np.zeros([num_master_groups, num_sub_in_grp], dtype='int32')
    ep_rets = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_master_groups)]
    ep_lens = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_master_groups)]

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros([horizon, num_master_groups, num_sub_in_grp], 'float32')
    vpreds = np.zeros([horizon, num_master_groups, num_sub_in_grp], 'float32')
    vpreds2 = np.zeros([horizon//macrolen, num_master_groups, num_sub_in_grp], 'float32')
    new = [[False for _ in range(num_sub_in_grp)] for _ in range(num_master_groups)]
    new = np.array(new)
    news = np.array([new for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    macro_acs = np.zeros([macro_horizon, num_master_groups, num_sub_in_grp], 'int32')
    macro_vpreds = np.zeros([macro_horizon, num_master_groups, num_sub_in_grp], 'float32')

    if recurrent:
        nlstm = args.nlstm
        state = np.zeros([num_master_groups, num_sub_in_grp, len(sub_policies), 2*nlstm], 
                dtype='float32')
        initial_state = state.copy()
        """
        states_tracker = np.array([state for _ in range(horizon)])
        states = np.zeros([horizon, num_master_groups, num_sub_in_grp, 2*nlstm], 
                dtype='float32')
        """

    prev_subpolicy = np.zeros([num_master_groups, num_sub_in_grp], 'int32')
    cur_subpolicy = np.zeros([num_master_groups, num_sub_in_grp], 'int32')
    
    while True:
        if t % macrolen == 0:
            # cur_subpolicy - num_master_groups * num_sub_in_grp
            # macro action selection
            prev_subpolicy = cur_subpolicy
            cur_subpolicy, macro_vpred = zip(*[policy.act(stochastic, ob[i]) 
                for i, policy in enumerate(policies)])

            # off policy-ness
            for i in range(num_master_groups):
                for j in range(num_sub_in_grp):
                    if np.random.uniform() < EPS:
                        cur_subpolicy[i][j] = np.random.randint(0, len(sub_policies))
                    #cur_subpolicy[i][j] = envs[i].envs[j].env.env.realgoal

            if args.force_subpolicy is not None:
                cur_subpolicy = [[args.force_subpolicy for _ in range(num_sub_in_grp)]
                        for _ in range(num_master_groups)]

        if t > 0 and t % horizon == 0:
            dicti = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "vpred2": vpreds2, 
                    "new" : news, "ac" : acs, "ep_rets" : (ep_rets), "ep_lens" : (ep_lens), 
                    "macro_ac" : macro_acs, "macro_vpred" : macro_vpreds}
            if recurrent:
                dicti["state"] = initial_state
                #dicti["state"]: states
            yield {key: np.copy(val) for key,val in dicti.items()}
            ep_rets = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_master_groups)]
            ep_lens = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_master_groups)]
            cur_ep_ret = np.zeros([num_master_groups, num_sub_in_grp], dtype='float32')
            cur_ep_len = np.zeros([num_master_groups, num_sub_in_grp], dtype='int32')

        # TODO: vectorize this
        # interaction with environment
        for i in range(num_master_groups):
            for j in range(num_sub_in_grp):
                if not recurrent:
                    ac[i][j], vpred[i][j] = sub_policies[cur_subpolicy[i][j]].act(stochastic, 
                            [ob[i][j]])
                else:
                    """
                    ac[i][j], vpred[i][j], state[i][j][cur_subpolicy[i][j]] = \
                            sub_policies[cur_subpolicy[i][j]].act(stochastic, [ob[i][j]], 
                            [state[i][j]][cur_subpolicy[i][j]])
                    """
                    for k, sub_policy in enumerate(sub_policies):
                        _ac, _vpred, _state = sub_policy.act(stochastic, [ob[i][j]], 
                                [state[i][j][k]], [float(new[i][j])]) 
                        state[i][j][k] = _state[0]
                        if k == cur_subpolicy[i][j]: 
                            ac[i][j] = _ac
                            vpred[i][j] = _vpred
                        l = t % horizon
                        if l % macrolen == 0 and prev_subpolicy[i][j] == k:
                            vpreds2[l//macrolen][i][j] = _vpred

                k = t % horizon
                if k % macrolen == 0:# and k//macrolen != horizon//macrolen:
                    if not recurrent:
                        _, vpreds2[k//macrolen][i][j] = \
                                sub_policies[prev_subpolicy[i][j]].act(stochastic, [ob[i][j]])
                    """
                    else:
                        _, vpreds2[k//macrolen][i][j], _ = \
                                sub_policies[prev_subpolicy[i][j]].act(stochastic, 
                                        [ob[i][j]], [state[i][j][prev_subpolicy[i][j]]], 
                                        [int(new[i][j])])
                    """

        i = t % horizon
        obs[i] = ob # current observation (t)
        vpreds[i] = vpred # current state's (t)  baseline
        news[i] = new # done (t-1 -> t transition)
        acs[i] = ac # current action (t)
        """
        if recurrent:
            _state = np.zeros([num_master_groups, num_sub_in_grp, 2*nlstm])
            for j in range(num_master_groups):
                for k in range(num_sub_in_grp):
                    _state[j][k] = state[j][k][cur_subpolicy[j][k]]
            states[i] = _state
        """
        if t % macrolen == 0:
            macro_acs[int(i/macrolen)] = cur_subpolicy
            macro_vpreds[int(i/macrolen)] = macro_vpred

        ob, rew, _new, info = zip(*[env.step(ac[i]) for i, env in enumerate(envs)])
        rews[i] = rew

        # TODO: replay - render the environment every few steps
        if replay:
            #if t % macrolen == 0:
            #    print(cur_subpolicy)
            envs[0].envs[0].render()
            time.sleep(0.05)

        cur_ep_ret += rew
        cur_ep_len += 1
        new = np.logical_or(new, _new) 
        # resets
        for i in range(num_master_groups):
            for j in range(num_sub_in_grp):
                if new[i][j] and ((t+1) % macrolen == 0):
                    ep_rets[i][j].append(cur_ep_ret[i][j])
                    ep_lens[i][j].append(cur_ep_len[i][j])
                    cur_ep_ret[i][j] = 0
                    cur_ep_len[i][j] = 0
                    ob[i][j] = envs[i].envs[j].reset()
                    new[i][j] = False
        t += 1

def add_advantage_macro(seg, macrolen, gamma, lam):
    group_shape = list(seg["new"][0].shape)
    new = np.append(seg["new"][0::macrolen], np.zeros(group_shape, dtype='int32')) 
    vpred = np.append(seg["macro_vpred"], np.zeros(group_shape, dtype='float32')) 
    T = int(len(seg["rew"])/macrolen)
    seg["macro_adv"] = gaelam = np.empty([T]+group_shape, 'float32')
    # macro rewards for master
    rew = np.sum(seg["rew"].reshape(-1, macrolen, group_shape[0], group_shape[1]), axis=1)
    lastgaelam = np.zeros(group_shape, dtype='float32') 
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        currentnonterminal = 1-new[t]
        #delta = currentnonterminal * delta
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["macro_tdlamret"] = seg["macro_adv"] + seg["macro_vpred"]
    seg["macro_ob"] = seg["ob"][0::macrolen]

# TODO: terminal states logic for the subpolicies
def prepare_allrolls(allrolls, macrolen, gamma, lam, num_subpolicies, recurrent=False):
    test_seg = allrolls[0]
    group_shape = list(test_seg["new"][0].shape)
    # calculate advantages
    new = np.append(test_seg["new"], np.zeros(group_shape, dtype='int32')) 
    vpred = np.append(test_seg["vpred"], np.zeros(group_shape, dtype='float32')) 
    vpred2 = np.append(test_seg["vpred2"], np.zeros(group_shape, dtype='float32')) 
    T = len(test_seg["rew"])
    test_seg["adv"] = gaelam = np.empty([T]+group_shape, 'float32')
    rew = test_seg["rew"]
    lastgaelam = np.zeros(group_shape, dtype='float32') 
    for t in reversed(range(T)):
        target_vpred = vpred2[(t+1)//macrolen] if (t+1)%macrolen == 0 else vpred[t+1]
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * target_vpred * nonterminal - vpred[t]
        currentnonterminal = 1-new[t]
        #delta = currentnonterminal * delta
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    test_seg["tdlamret"] = test_seg["adv"] + test_seg["vpred"]

    split_test = split_segments(test_seg, macrolen, num_subpolicies, recurrent=recurrent)
    return split_test

# TODO: make parallel
def split_segments(seg, macrolen, num_subpolicies, recurrent=False):
    group_shape = seg["new"][0].shape
    num_master_groups, num_sub_in_grp = group_shape
    if not recurrent:
        subpol_counts = np.zeros([num_subpolicies], dtype='int32')
        for macro_ac in seg["macro_ac"]:
            for i in range(num_master_groups):
                for j in range(num_sub_in_grp):
                    subpol_counts[macro_ac[i][j]] += macrolen
        subpols = []
        for i in range(num_subpolicies):
            obs = np.array([seg["ob"][0][0][0] for _ in range(subpol_counts[i])])
            advs = np.zeros(subpol_counts[i], 'float32')
            tdlams = np.zeros(subpol_counts[i], 'float32')
            news = np.zeros(subpol_counts[i], 'int32')
            acs = np.array([seg["ac"][0][0][0] for _ in range(subpol_counts[i])])
            subpols.append({"ob": obs, "adv": advs, "tdlamret": tdlams, "ac": acs, 
                "new": news}) 
        subpol_counts = []
        for i in range(num_subpolicies):
            subpol_counts.append(0)
        for i in range(len(seg["ob"])):
            for j in range(num_master_groups):
                for k in range(num_sub_in_grp):
                    mac = seg["macro_ac"][int(i/macrolen)][j][k]
                    subpols[mac]["ob"][subpol_counts[mac]] = seg["ob"][i][j][k]
                    subpols[mac]["adv"][subpol_counts[mac]] = seg["adv"][i][j][k]
                    subpols[mac]["tdlamret"][subpol_counts[mac]] = seg["tdlamret"][i][j][k]
                    subpols[mac]["ac"][subpol_counts[mac]] = seg["ac"][i][j][k]
                    subpols[mac]["new"][subpol_counts[mac]] = seg["new"][i][j][k]
                    subpol_counts[mac] += 1
    else:
        subpols = []
        horizon = len(seg["ob"])
        T = num_master_groups*num_sub_in_grp*horizon
        num_env = num_master_groups*num_sub_in_grp
        for i in range(num_subpolicies):
            obs = np.array([[seg["ob"][0][0][0] for _ in range(horizon)] 
                for _ in range(num_env)])
            advs = np.zeros([num_env, horizon], 'float32')
            tdlams = np.zeros([num_env, horizon], 'float32')
            news = np.zeros([num_env, horizon], 'int32')
            masks = np.zeros([num_env, horizon], 'int32')
            acs = np.array([[seg["ac"][0][0][0] for _ in range(horizon)] 
                for _ in range(num_env)])
            states = np.array([seg["state"][0][0][0] for _ in range(num_env)])
            subpols.append({"ob": obs, "adv": advs, "tdlamret": tdlams, "ac": acs, 
                "new": news, "mask": masks, "state": states}) 

        for i in range(len(seg["ob"])):
            for j in range(num_master_groups):
                for k in range(num_sub_in_grp):
                    env_idx = (j*num_sub_in_grp)+k
                    #idx = ((j*num_sub_in_grp)+k)*horizon+i
                    mac = seg["macro_ac"][int(i/macrolen)][j][k]
                    subpols[mac]["ob"][env_idx][i] = seg["ob"][i][j][k]
                    subpols[mac]["adv"][env_idx][i] = seg["adv"][i][j][k]
                    subpols[mac]["tdlamret"][env_idx][i] = seg["tdlamret"][i][j][k]
                    subpols[mac]["ac"][env_idx][i] = seg["ac"][i][j][k]
                    subpols[mac]["new"][env_idx][i] = seg["new"][i][j][k]
                    subpols[mac]["mask"][env_idx][i] = 1
                    for l in range(num_subpolicies):
                        subpols[l]["new"][env_idx][i] = seg["new"][i][j][k]
                        subpols[l]["mask"][env_idx][i] = int(l == mac)

        for i in range(num_subpolicies):
            subpols[i]["ob"] = flatten_env_time_dims(subpols[i]["ob"])
            subpols[i]["adv"] = flatten_env_time_dims(subpols[i]["adv"])
            subpols[i]["tdlamret"] = flatten_env_time_dims(subpols[i]["tdlamret"])
            subpols[i]["ac"] = flatten_env_time_dims(subpols[i]["ac"])
            subpols[i]["new"] = flatten_env_time_dims(subpols[i]["new"])
            subpols[i]["mask"] = flatten_env_time_dims(subpols[i]["mask"])

    return subpols

def flatten_env_time_dims(arr):
    s = arr.shape
    return arr.reshape(s[0] * s[1], *s[2:])
