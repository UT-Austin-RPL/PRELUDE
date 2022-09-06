import numpy as np
from numbers import Number
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Tuple, Union, Callable, Optional
from tensorboard.backend.event_processing import event_accumulator


WRITE_TYPE = Union[int, Number, np.number, np.ndarray]

SAVE_REWARD_ITEMS = ['RewardSum', 'RewardVel', 'RewardPos', 'RewardAtt', 'RewardProgress', 'PenaltyJnt1', 'PenaltyJnt2', 'PenaltyJnt3', 'PenaltyAff', 'PenaltyAction']
SAVE_STATE_ITEMS = ['ErrorLin', 'ErrorYaw', 'ErrorPos', 'ErrorAtt', 'JointPow']
SAVE_ACTION_ITEMS = ['JointFR1', 'JointFR2', 'JointFR3', 'JointFL1', 'JointFL2', 'JointFL3', 'JointRR1', 'JointRR2', 'JointRR3', 'JointRL1', 'JointRL2', 'JointRL3']
SAVE_TERMIAL_ITEMS = ['Time', 'Success', 'Progress', 'Curriculum', 'Return']

class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer."""

    def __init__(self, writer: Any) -> None:
        super().__init__()
        self.writer = writer

    @abstractmethod
    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        """Specify how the writer is used to log data.

        :param str key: namespace which the input data tuple belongs to.
        :param int x: stands for the ordinate of the input data tuple.
        :param y: stands for the abscissa of the input data tuple.
        """
        pass

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        pass

    def restore_data(self) -> Tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """
        pass


class BasicLogger(BaseLogger):
    """A loggger that relies on tensorboard SummaryWriter by default to visualize \
    and log statistics.

    You can also rewrite write() func to use your own writer.

    :param SummaryWriter writer: the writer to log data.
    :param int train_interval: the log interval in log_train_data(). Default to 1.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data(). Default to 1000.
    :param int save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
    ) -> None:
        super().__init__(writer)
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_save_step = -1

    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        self.writer.add_scalar(key, y, global_step=x)


    def log_info(self, info, step, tagTrain = True):

        if tagTrain:
            strTag = 'Train'
            step = self.last_log_train_step
        else:
            strTag = 'Test'
            step = self.last_log_test_step

        for key, value in info.items():
            if key in SAVE_REWARD_ITEMS:
                self.writer.add_scalar('{}Reward/{}'.format(strTag, key), np.average(value), global_step=step)
            if key in SAVE_STATE_ITEMS:
                self.writer.add_scalar('{}State/{}'.format(strTag, key), np.average(value), global_step=step)
            if key in SAVE_ACTION_ITEMS:
                self.writer.add_scalar('{}Action/{}'.format(strTag, key), np.max(np.absolute(value)), global_step=step)

        if 'Done' in info:
            numEpisode = np.max([np.sum(info['Done']), 1])

            for key, value in info.items():
                if key in SAVE_TERMIAL_ITEMS:
                    self.writer.add_scalar('{}Episode/{}'.format(strTag, key), np.sum(value)/(numEpisode+0.0001), global_step=step)


    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew" and "len" keys.
        """
        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            if step - self.last_log_train_step >= self.train_interval:
                self.write("TrainInfo/n/ep", step, collect_result["n/ep"])
                self.write("TrainInfo/rew", step, collect_result["rew"])
                self.write("TrainInfo/len", step, collect_result["len"])
                self.log_info(collect_result['infos'], step, tagTrain = True)
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew", "rew_std", "len",
            and "len_std" keys.
        """
        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:
            self.write("TestInfo/rew", step, rew)
            self.write("TestInfo/len", step, len_)
            self.write("TestInfo/rew_std", step, rew_std)
            self.write("TestInfo/len_std", step, len_std)
            self.log_info(collect_result['infos'], step, tagTrain = False)
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            for k, v in update_result.items():
                self.write(k, step, v)
            self.last_log_update_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            self.write("Save/epoch", epoch, epoch)
            self.write("Save/env_step", env_step, env_step)
            self.write("Save/gradient_step", gradient_step, gradient_step)

    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("Save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("Save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("Save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step


class LazyLogger(BasicLogger):
    """A loggger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        """The LazyLogger writes nothing."""
        pass
