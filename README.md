# Meta-Learning Shared Hierarchies

GPU implementation of [Meta-Learning Shared Hierarchies](https://s3-us-west-2.amazonaws.com/openai-assets/MLSH/mlsh_paper.pdf), based on [OpenAI's MPI based implementation](https://github.com/openai/mlsh)

This branch is non-experimental. It was created as an attempt to replicate the OpenAI implementation only.

##### Installation

```
Add the following to your .bash_profile (replace ... with path to directory) and source it:
export PYTHONPATH=$PYTHONPATH:/.../mlsh-gpu/rl-algs;
```

##### How to run

```
python3 main.py --task [task_name] --num_subs [num_subs] --macro_duration [macro_duration] --num_rollouts [num_rollouts] --warmup_time [warmup_time] --train_time [train_time] --num_master_grp [num_master_grp] --num_sub_batches [num_sub_batches] --num_sub_in_grp [num_sub_in_grp] --vfcoeff [vfcoeff] --entcoeff [entcoeff] --master_lr [master_lr] --sub_lr [sub_lr] --replay [replay] [savename] 
```

##### Note

DEVELOPMENT IN PROGRESS - see code for pending TODOs
