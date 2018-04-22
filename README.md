# Meta-Learning Shared Hierarchies

GPU implementation of [Meta-Learning Shared Hierarchies](https://s3-us-west-2.amazonaws.com/openai-assets/MLSH/mlsh_paper.pdf), based on [OpenAI's MPI based implementation](https://github.com/openai/mlsh)


##### Installation

```
Add the following to your .bash_profile (replace ... with path to directory) and source it:
export PYTHONPATH=$PYTHONPATH:/.../mlsh-gpu/rl-algs;
```

##### How to run

```
python3 main.py --task [task_name] --num_subs [num_subs] --macro_duration [macro_duration] --num_rollouts [num_rollouts] --warmup_time [warmup_time] --train_time [train_time] --num_master_grp [num_master_grp] --num_sub_batches [num_sub_batches] --num_sub_in_grp [num_sub_in_grp] --vfcoeff [vfcoeff] --entcoeff [entcoeff] --master_lr [master_lr] --sub_lr [sub_lr] --replay [replay] [savename] 
```

##### To run on AWS
1. Upgrade conda, do `conda update -n base conda`
2. Do `source activate tensorflow_p36`
3. Upgrade pip, do `pip install --upgrade pip`
4. Do `git clone https://github.com/ysaibhargav/mlsh-gpu.git`
5. Do `export PYTHONPATH=$PYTHONPATH:/home/ubuntu/mlsh-gpu/rl-algs:/home/ubuntu/mlsh-gpu/test_envs;`
6. Use the env.yml file to create a conda environment, do `conda env create -f mlsh-gpu/env.yml` 
7. Do `conda activate mlsh_p36`
8. At the time of writing this document, the class `DummyVecEnv` is used from the GitHub repo for the `baselines` package which is not yet released. We therefore install from source. Do `pip uninstall baselines`.
9. Do `git clone https://github.com/openai/baselines.git`
10. Do `cd baselines; python setup.py install`
11. Run `main.py` 

##### Note

DEVELOPMENT IN PROGRESS - see code for pending TODOs
