# Meta-Learning Shared Hierarchies

GPU implementation of [Meta-Learning Shared Hierarchies](https://s3-us-west-2.amazonaws.com/openai-assets/MLSH/mlsh_paper.pdf), based on [OpenAI's MPI based implementation](https://github.com/openai/mlsh)


##### Installation

```
Add to your .bash_profile (replace ... with path to directory):
export PYTHONPATH=$PYTHONPATH:/.../mlsh-gpu/rl-algs;
```

##### How to run

```
python3 main.py --task [task_name] --num_subs [num_subs] --macro_duration [macro_duration] --num_rollouts [num_rollouts] --warmup_time [warmup_time] --train_time [train_time] --replay [replay] [savename] 
```

##### Note

DEVELOPMENT IN PROGRESS - see code for pending TODOs
