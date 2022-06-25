import numpy as np
import torch as th


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class BaseBuffer(object):
    def __init__(self, obs, buffer_size, device):

        self.buffer_size = buffer_size
        self.device = device

        self.pos = 0
        self.full = False
        self.device = device

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError