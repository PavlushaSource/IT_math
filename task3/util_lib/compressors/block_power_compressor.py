import time

from .common_compressor import SVDCompressor
from .representation_BMP import SVD
from abc import ABC
import numpy as np


class BlockPowerSVDCompressor(SVDCompressor, ABC):
    def __init__(self, duration: float):
        self.duration = duration
        self.epsilon = 1e-8

    def _compress_channel(self, channel: np.ndarray, k: int):
        u = np.zeros((channel.shape[0], k))
        s = np.zeros(k)
        vh = np.zeros((channel.shape[1], k))

        err = self.epsilon + 1
        time_end = time.time() * 1000 + self.duration
        while time.time() * 1000 < time_end or err > self.epsilon:
            q, _ = np.linalg.qr(np.dot(channel, vh))
            u = q[:, :k]

            q, r = np.linalg.qr(np.dot(channel.T, u))
            vh = q[:, :k]
            s = r[:k, :k]

            err = np.linalg.norm(np.dot(channel, vh) - np.dot(u, s))

        return SVD(u, np.diag(s).astype(np.float32), vh.T)
