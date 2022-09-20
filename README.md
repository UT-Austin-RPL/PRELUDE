# Learning to Walk by Steering: Perceptive Quadrupedal Locomotion in Dynamic Environments
[Mingyo Seo](https://mingyoseo.com), [Ryan Gupta](https://sites.utexas.edu/hcrl/people/), [Yifeng Zhu](https://www.cs.utexas.edu/~yifengz), [Alexy Skoutnev](https://alexyskoutnev.github.io/alexyskoutnev-github.io/index.html), [Luis Sentis](https://sites.google.com/view/lsentis), [Yuke Zhu](https://www.cs.utexas.edu/~yukez)

[Project](https://ut-austin-rpl.github.io/PRELUDE) | [arXiv](http://arxiv.org/abs/2209.09233)

![intro](pipeline.png)

## Introduction
We tackle the problem of perceptive locomotion in dynamic environments. In this problem, a quadruped robot must exhibit robust and agile walking behaviors in response to environmental clutter and moving obstacles. We present a hierarchical learning framework, named PRELUDE, which decomposes the problem of perceptive locomotion into high-level decision making to predict navigation commands and low-level gait generation to realize the target commands. In this framework, we train the high-level navigation controller with imitation learning on human demonstrations collected on a steerable cart and the low-level gait controller with reinforcement learning (RL). Our method is, therefore, able to acquire complex navigation behaviors from human supervision and discover versatile gaits from trial and error. We demonstrate the effectiveness of our approach in simulation and with hardware experiments. Compared to state-of-the-art RL baselines, our method outperforms them by 38.6% in average distance traversed.

If you find our work useful in your research, please consider [citing](#citing).

## Dependencies
- Python 3.8.5 (recommended)
- [Robomimic](https://github.com/ARISE-Initiative/robomimic/)
- [Tianshou](https://github.com/thu-ml/tianshou/)
- [PyBullet](https://github.com/bulletphysics/bullet3/)
- [PyTorch](https://github.com/pytorch/pytorch)

## Installation
Install the environments and dependencies by running the following commands.
```
pip3 install -e .
```
You need to locate asset files to `./data/*`. These asset files are found [here](https://utexas.box.com/s/oa5c39blv9ma4h4lkdkv84n5zj3mxcg5).

## Creating a demo dataset for Navigation Controller

For collecting human demonstration data for Navigation Controller, please use the following commands. You may need a Spacemouse.
```
python3 scripts/demo_nav.py --env_type=ENV_TYPE --demo_name=DEMO_NAME
```
You may be able to specify the difficulty of environments by changing ENV_TYPE. Collected data would be saved in `./save/data_sim/DEMO_NAME` as `pickle` files. Rendering videos and extra logs would be saved in `./save/raw_sim/DEMO_NAME`.

To convert the collected data into `hdf5` dataset file, please use the following commands. The converted dataset would be saved in `PATH_TO_TARGET_FILE`. 
```
python3 scripts/utils/convert_dataset.py --folder=PATH_TO_DATA_FOLDER --demo_path=PATH_TO_TARGET_FILE
```
Then, please run the following commands to split the dataset for training and evaluation. The script would overwrite the split dataset would on the original dataset file.
```
python3 scripts/utils/split_train_val.py --dataset=PATH_TO_TARGET_FILE
```
Dataset files consist of sequences of the following data structure.
```
hdf5 dataset
├── agentview_rgb: 212x120x3 array
├── agentview_depth: 212x120x1 array
├── yaw: 2D value
├── actions: 2D value
├── dones: 1D value
└── rewards: 1D value (not used)
```

## Training
For training the Gait Controller, please use the following commands. Trained files would be saved in `./save/rl_checkpoint/gait/GAIT_POLICY`
```
python3 scripts/train_gait.py --gait_policy=GAIT_POLICY
```

For training Navigation Controller, please use the following commands. You need to create or download ([link](https://utexas.box.com/s/vuneto210i5o5c8vi09cxt49dta2may3)) an `hdf5`-format Dataset file for training. Trained files would be saved in `./save/bc_checkpoint`.
```
python3 scripts/train_nav.py
```

## Evaluation
You should locate pre-trained data to `./save/*`. These pre-trained data would be released later.

For evaluating Gait Controller only, please use the following commands. The checkpoints of the Gait Controller at `./save/rl_checkpoint/gait/RL_POLICY` would be loaded.
```
python3 scripts/eval_gait.py --gait_policy=GAIT_POLICY
```

To evaluate PRELUDE with both Gait Controller and Navigation Controller, please use the following commands. The checkpoints of the Navigation Controller at `./save/bc_checkpoint/NAV_POLICY` would be loaded.
```
python3 scripts/eval_nav.py --gait_policy=GAIT_POLICY --nav_policy=NAV_POLICY --gait_policy=GAIT_POLICY
```


## Dataset and pre-trained models
We provide our demonstration dataset in simulation environments ([link](https://utexas.box.com/s/vuneto210i5o5c8vi09cxt49dta2may3)) and trained models of the Navigation Controller ([link](https://utexas.box.com/s/l6n5unyswuol4gxwam552u1jkogbaakq)) and the Gait Controller ([link](https://utexas.box.com/s/uv41n7550t1ao7wv0io0er2s8r2ivu2x)).


## Implementation Details
Please see [this page](implementation.md) for more information about our implementation details, including training procedures and hyperparameters.

## Citing
```
@inproceedings{seo2022learning,
   title={Learning to Walk by Steering: Perceptive Quadrupedal Locomotion in Dynamic Environments},
   author={Seo, Mingyo and Gupta, Ryan and Zhu, Yifeng and Skoutnev, Alexy and Sentis, Luis and Zhu, Yuke},
   booktitle={arXiv preprint arXiv:2209.09233},
   year={2022}
}
```
