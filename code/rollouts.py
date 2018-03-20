import numpy as np
import math
import time
import pdb

def traj_segment_generator(policies, sub_policies, envs, macrolen, horizon, 
        num_sub_in_grp, stochastic, args):
    num_policies = len(policies)
    replay = args.replay
    t = 0
    # ac - num_policies * num_sub_in_grp
    ac = [[envs[0].action_space.sample() for _ in range(num_sub_in_grp)] for env in envs]
    ac = np.array(ac, dtype='int32')
    vpred = np.zeros([num_policies, num_sub_in_grp], dtype='float32')
    new = [[True for _ in range(num_sub_in_grp)] for _ in range(num_policies)] 
    rew = [[0.0 for _ in range(num_sub_in_grp)] for _ in range(num_policies)] 
    # ob - num_policies * snum_sub_in_grp * state_dims
    ob = [env.reset() for env in envs]
    #cur_subpolicy = [0 for _ in range(num_policies)]
    #macro_vpred = [0 for _ in range(num_policies)]
    macro_horizon = math.ceil(horizon/macrolen)

    cur_ep_ret = np.zeros([num_policies, num_sub_in_grp], dtype='float32')
    cur_ep_len = np.zeros([num_policies, num_sub_in_grp], dtype='int32')
    ep_rets = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_policies)]
    ep_lens = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_policies)]

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros([horizon, num_policies, num_sub_in_grp], 'float32')
    vpreds = np.zeros([horizon, num_policies, num_sub_in_grp], 'float32')
    news = np.zeros([horizon, num_policies, num_sub_in_grp], 'int32')
    acs = np.array([ac for _ in range(horizon)])
    macro_acs = np.zeros([macro_horizon, num_policies, num_sub_in_grp], 'int32')
    macro_vpreds = np.zeros([macro_horizon, num_policies, num_sub_in_grp], 'float32')

    while True:
        if t % macrolen == 0:
            cur_subpolicy, macro_vpred = zip(*[policy.act(stochastic, ob[i]) 
                for i, policy in enumerate(policies)])

            for i in range(num_policies):
                for j in range(num_sub_in_grp):
                    if np.random.uniform() < 0.1:
                        cur_subpolicy[i][j] = np.random.randint(0, len(sub_policies))
            if args.force_subpolicy is not None:
                cur_subpolicy = [[args.force_subpolicy for _ in range(num_sub_in_grp)]
                        for _ in range(num_policies)]

        for i in range(num_policies):
            for j in range(num_sub_in_grp):
                ac[i][j], vpred[i][j] = sub_policies[cur_subpolicy[i][j]].act(stochastic, 
                        [ob[i][j]])
        
        """
        ac, vpred = zip(*[sub_policies[cur_subpolicy[i]].act(stochastic, ob[i]) for i 
                in range(num_policies)])
        """

        if t > 0 and t % horizon == 0:
            dicti = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news, 
                    "ac" : acs, "ep_rets" : ep_rets, "ep_lens" : ep_lens, 
                    "macro_ac" : macro_acs, "macro_vpred" : macro_vpreds}
            yield {key: np.copy(val) for key,val in dicti.items()}
            ep_rets = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_policies)]
            ep_lens = [[[] for _ in range(num_sub_in_grp)] for _ in range(num_policies)]

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        if t % macrolen == 0:
            macro_acs[int(i/macrolen)] = cur_subpolicy
            macro_vpreds[int(i/macrolen)] = macro_vpred

        ob, rew, new, info = zip(*[env.step(ac[i]) for i, env in enumerate(envs)])
        rews[i] = rew

        # TODO: replay
        """
        if replay:
            if len(ep_rets) == 0:
                # if x % 5 == 0:
                env.render()
                    # print(info)
        """

        cur_ep_ret += rew
        cur_ep_len += 1
        for i in range(num_policies):
            for j in range(num_sub_in_grp):
                if new[i][j] and ((t+1) % macrolen == 0):
                    ep_rets[i][j].append(cur_ep_ret[i][j])
                    ep_lens[i][j].append(cur_ep_len[i][j])
                    cur_ep_ret[i][j] = 0
                    cur_ep_len[i][j] = 0
                    ob[i][j] = envs[i].envs[j].reset()
        t += 1

def add_advantage_macro(seg, macrolen, gamma, lam):
    group_shape = list(seg["new"][0].shape)
    new = np.append(seg["new"][0::macrolen], np.zeros(group_shape, dtype='int32')) 
    vpred = np.append(seg["macro_vpred"], np.zeros(group_shape, dtype='float32')) 
    T = int(len(seg["rew"])/macrolen)
    seg["macro_adv"] = gaelam = np.empty([T]+group_shape, 'float32')
    # TODO: double check reshaping in high dims
    rew = np.sum(seg["rew"].reshape(-1, macrolen, group_shape[0], group_shape[1]), axis=1)
    lastgaelam = np.zeros(group_shape, dtype='float32') 
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["macro_tdlamret"] = seg["macro_adv"] + seg["macro_vpred"]
    seg["macro_ob"] = seg["ob"][0::macrolen]

def prepare_allrolls(allrolls, macrolen, gamma, lam, num_subpolicies):
    """
    for i in range(len(allrolls) - 1):
        for key,value in allrolls[i + 1].items():
            allrolls[0][key] = np.append(allrolls[0][key], value, axis=0)
    """
    test_seg = allrolls[0]
    group_shape = list(test_seg["new"][0].shape)
    # calculate advantages
    new = np.append(test_seg["new"], np.zeros(group_shape, dtype='int32')) 
    vpred = np.append(test_seg["vpred"], np.zeros(group_shape, dtype='float32')) 
    T = len(test_seg["rew"])
    test_seg["adv"] = gaelam = np.empty([T]+group_shape, 'float32')
    rew = test_seg["rew"]
    lastgaelam = np.zeros(group_shape, dtype='float32') 
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    test_seg["tdlamret"] = test_seg["adv"] + test_seg["vpred"]

    split_test = split_segments(test_seg, macrolen, num_subpolicies)
    return split_test

def split_segments(seg, macrolen, num_subpolicies):
    group_shape = seg["new"][0].shape
    num_policies, num_sub_in_grp = group_shape
    subpol_counts = np.zeros([num_subpolicies], dtype='int32')
    for macro_ac in seg["macro_ac"]:
        for i in range(num_policies):
            for j in range(num_sub_in_grp):
                subpol_counts[macro_ac[i][j]] += macrolen
    subpols = []
    for i in range(num_subpolicies):
        obs = np.array([seg["ob"][0][0][0] for _ in range(subpol_counts[i])])
        advs = np.zeros(subpol_counts[i], 'float32')
        tdlams = np.zeros(subpol_counts[i], 'float32')
        acs = np.array([seg["ac"][0][0][0] for _ in range(subpol_counts[i])])
        subpols.append({"ob": obs, "adv": advs, "tdlamret": tdlams, "ac": acs})
    subpol_counts = []
    for i in range(num_subpolicies):
        subpol_counts.append(0)
    for i in range(len(seg["ob"])):
        for j in range(num_policies):
            for k in range(num_subpolicies):
                mac = seg["macro_ac"][int(i/macrolen)][j][k]
                subpols[mac]["ob"][subpol_counts[mac]] = seg["ob"][i][j][k]
                subpols[mac]["adv"][subpol_counts[mac]] = seg["adv"][i][j][k]
                subpols[mac]["tdlamret"][subpol_counts[mac]] = seg["tdlamret"][i][j][k]
                subpols[mac]["ac"][subpol_counts[mac]] = seg["ac"][i][j][k]
                subpol_counts[mac] += 1
    return subpols
