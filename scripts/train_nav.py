import argparse
import json
import numpy as np
import os
import sys

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils.log_utils import PrintLogger
from robomimic.envs.env_base import EnvType
from robomimic.utils.dataset import SequenceDataset

from path import *

## Load demostration data
def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.
    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)
    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    filter_by_attribute = config.train.hdf5_filter_key

    # load the dataset into memory
    assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
    train_filter_by_attribute = "train"
    valid_filter_by_attribute = "valid"
    if filter_by_attribute is not None:
        train_filter_by_attribute = "{}_{}".format(filter_by_attribute, train_filter_by_attribute)
        valid_filter_by_attribute = "{}_{}".format(filter_by_attribute, valid_filter_by_attribute)

    def get_dataset(filter_by_attribute):
        return SequenceDataset(
            hdf5_path=config.train.data,
            obs_keys=obs_keys,
            dataset_keys=config.train.dataset_keys,
            load_next_obs=False,
            frame_stack=1,
            seq_length=config.train.seq_length,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=config.train.goal_mode,
            hdf5_cache_mode=config.train.hdf5_cache_mode,
            hdf5_use_swmr=config.train.hdf5_use_swmr,
            hdf5_normalize_obs=config.train.hdf5_normalize_obs,
            filter_by_attribute=filter_by_attribute,
        )

    train_dataset = get_dataset(train_filter_by_attribute)
    valid_dataset = get_dataset(valid_filter_by_attribute)


    return train_dataset, valid_dataset


## Train Navigation Controller
def train(config, device):
    """
    Train a model using the algorithm.
    """

    # Configuration
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # Configure meta data
    env_meta = {
        "env_name": "quadruped-nav",
        "type": EnvType.GYM_TYPE,
        "env_kwargs": {}
    }
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    # BC Model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device
    )
    print("\n============= Model Summary =============")
    print(model)

    # Load data set
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    trainset, validset = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,       # no custom sampling logic (uniform sampling)
        batch_size=config.train.batch_size,     # batches of size 100
        shuffle=True,
        num_workers=config.train.num_data_workers,
        drop_last=True# don't provide last batch in dataset pass if it's less than 100 in size
    )

    valid_sampler = validset.get_dataset_sampler()
    valid_loader = DataLoader(
        dataset=validset,
        sampler=valid_sampler,
        batch_size=config.train.batch_size,
        shuffle=(valid_sampler is None),
        num_workers=0,
        drop_last=True
        )

    # Train
    best_valid_loss = None
    best_training_loss = None
    for epoch in range(1, config.train.num_epochs + 1):
        should_save_ckpt = False
        train_step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=config.experiment.epoch_every_n_steps)
        model.on_epoch_end(epoch)

        print("Train Epoch {}: Loss {}".format(epoch, train_step_log["Loss"]))

        if best_training_loss is None or train_step_log["Loss"] < best_training_loss:
            should_save_ckpt = True
            epoch_ckpt_name = "model_best_training"
            print("Best Model Loss: Loss {}".format(train_step_log["Loss"]))
        
        with torch.no_grad():
            valid_step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=config.experiment.validation_epoch_every_n_steps)
        valid_check = "Loss" in valid_step_log
        if valid_check and (best_valid_loss is None or (valid_step_log["Loss"] <= best_valid_loss)):
            best_valid_loss = valid_step_log["Loss"]
            valid_epoch_ckpt_name = "model_best_validation"
            should_save_ckpt = True
            print("Best Validation Loss: Loss {}".format(valid_step_log["Loss"]))

        if should_save_ckpt:
            print("Saving checkpoint")
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        if epoch >= config.experiment.rollout.warmstart:
            pass
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default='bcrnn',
        help="path to a config json that will be used to override the default settings. \
              For example, --config=CONFIG.json will load a .json config file at ./config/nav/CONFIG.json.\
              If omitted, default settings are used. This is the preferred way to run experiments.")
    args = parser.parse_args()

    ext_cfg = json.load(open('{}/nav/{}.json'.format(PATH_CONFIG, args.config), 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    model = train(config, device)