import time

from .common_compressor import SVDCompressor
from .representation_BMP import SVD
from abc import ABC
import numpy as np


class PowerSVDCompressor(SVDCompressor, ABC):
    def __init__(self, duration: float):
        self.duration = duration
        self.epsilon = 1e-8

    def _power_svd(self, channel: np.ndarray, duration: float):
        time_end = time.time() * 1000 + duration
        mu, sigma = 0, 1
        x = np.random.normal(mu, sigma, size=channel.shape[1])
        b = channel.T.dot(channel)
        while True:
            new_x = b.dot(x)
            if np.allclose(new_x, x, self.epsilon):
                break
            x = new_x / np.linalg.norm(new_x)
            if time.time() * 1000 >= time_end:
                break

        v = x / np.linalg.norm(x)
        sigma = np.linalg.norm(channel.dot(v))
        u = channel.dot(v) / sigma
        return np.reshape(u, (channel.shape[0], 1)), sigma, np.reshape(v, (channel.shape[1], 1))

    def _compress_channel(self, channel: np.ndarray, k: int):
        rank = np.linalg.matrix_rank(channel)
        ut = np.zeros((channel.shape[0], 1))
        st = []
        vht = np.zeros((channel.shape[1], 1))

        single_duration = self.duration / rank

        for i in range(rank):
            u, sigma, v = self._power_svd(channel, single_duration)
            ut = np.hstack((ut, u))
            st.append(sigma)
            vht = np.hstack((vht, v))
            channel = channel - u.dot(v.T).dot(sigma)

        ut = ut[:, 1:]
        vht = vht[:, 1:]
        return SVD(ut[:, :k], np.array(st)[:k], vht.T[:k, :])
