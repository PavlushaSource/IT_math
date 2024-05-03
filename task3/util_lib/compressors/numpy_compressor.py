from .common_compressor import SVDCompressor
from abc import ABC
from .representation_BMP import SVD
import numpy as np


class NumpySVDCompressor(SVDCompressor, ABC):
    def _compress_channel(self, channel: np.ndarray, k: int):
        u, s, vh = np.linalg.svd(channel, full_matrices=False)
        return SVD(u[:, :k], s[:k], vh[:k, :])
