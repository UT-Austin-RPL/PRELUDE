import numpy as np
from numbers import Number
from typing import Any, Tuple, Union, Callable, Optional
import wandb

WRITE_TYPE = Union[int, Number, np.number, np.ndarray]

SAVE_REWARD_ITEMS = [   "Reward Tracking",        "Reward Balance",      "Reward Gait",
                        "Reward Energy",          "Reward Badfoot",      "Reward Footcontact"]
SAVE_STATE_ITEMS =  [   "Height",       "Jonit Power"  ,     "Contact",
                        "Roll Rate",     "Pitch Rate",    "Yaw Rate",      
                        "Error XY",      "Error Yaw",
                        "Drift"]
SAVE_ACTION_ITEMS = [   ]
SAVE_TERMIAL_ITEMS = [  ]


class WandbLogger():
    """
    WandB logger for training Gait Controller
    """

    def __init__(
        self,
        project: str= "project",
        task: str= "task",
        path: str= "./log.dat",
        update_interval: int = 1000,
        actor: Any=None,
        critic: Any=None,
        reward_config = {},
        ppo_config = {},
        experiment_config = {},
        simulation_config = {},
    ) -> None:

        self.id = wandb.util.generate_id()
        self.writer = wandb.init(id=self.id, resume="allow",
                                project=project,
                                job_type=task,
                                config=ppo_config,
                                save_code=path,
                                dir=path,
                                sync_tensorboard=False
                                )
        self.writer.config.update(reward_config)
        self.writer.config.update(experiment_config)
        self.writer.config.update(simulation_config)

        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_save_step = -1
        self.num_episode = 0

        self.writer.watch(actor, log_freq=1000)
        self.writer.watch(critic, log_freq=1000)

    def log_train_data(self, collect_result: dict, step: int) -> None:

        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            collect_result["rew_std"] = collect_result["rews"].std()
            collect_result["len_std"] = collect_result["lens"].std()

            if step - self.last_log_train_step >= self.update_interval:
                dicLogInfo = {}

                dicLogInfo["train/episode/Reward"] = collect_result["rew"]
                dicLogInfo["train/episode/Length"] = collect_result["len"]
                dicLogInfo["train/episode/Reward Std"] = collect_result["rew_std"]
                dicLogInfo["train/episode/Length Std"] = collect_result["len_std"]

                for key, value in collect_result["infos"].items():
                    if key in SAVE_REWARD_ITEMS:
                        dicLogInfo["train/reward/{}".format(key)] = np.average(value)
                    if key in SAVE_STATE_ITEMS:
                        dicLogInfo["train/state/{}".format(key)] = np.average(value)
                    if key in SAVE_ACTION_ITEMS:
                        dicLogInfo["train/action/{}".format(key)] = np.max(np.absolute(value))
                    if key in SAVE_TERMIAL_ITEMS:
                        dicLogInfo["train/episode/{}".format(key)] = np.average(value)

                self.writer.log(dicLogInfo, step=step)
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:

        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            collect_result["rew_std"] = collect_result["rews"].std()
            collect_result["len_std"] = collect_result["lens"].std()

            if step - self.last_log_train_step >= self.update_interval:
                dicLogInfo = {}

                dicLogInfo["test/episode/Reward"] = collect_result["rew"]
                dicLogInfo["test/episode/Length"] = collect_result["len"]
                dicLogInfo["test/episode/Reward Std"] = collect_result["rew_std"]
                dicLogInfo["test/episode/Length Std"] = collect_result["len_std"]

                for key, value in collect_result["infos"].items():
                    if key in SAVE_REWARD_ITEMS:
                        dicLogInfo["test/reward/{}".format(key)] = np.average(value)
                    if key in SAVE_STATE_ITEMS:
                        dicLogInfo["test/state/{}".format(key)] = np.average(value)
                    if key in SAVE_ACTION_ITEMS:
                        dicLogInfo["test/action/{}".format(key)] = np.max(np.absolute(value))
                    if key in SAVE_TERMIAL_ITEMS:
                        dicLogInfo["test/episode/{}".format(key)] = np.average(value)

                self.writer.log(dicLogInfo, step=step)
                self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            self.writer.log(update_result)
            self.last_log_update_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if save_checkpoint_fn and env_step - self.last_save_step >= 1:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            dicLogInfo = {}
            dicLogInfo["save/Epoch"] = epoch
            dicLogInfo["save/EnvStep"] = env_step
            dicLogInfo["save/GradientStep"] = gradient_step
            self.writer.log(dicLogInfo)

    def restore_data(self) -> Tuple[int, int, int]:

        return None

