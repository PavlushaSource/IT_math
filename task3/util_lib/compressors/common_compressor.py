import math
import os
from abc import ABC, abstractmethod
from PIL import Image

import numpy as np
from .representation_BMP import SVD, ImgCompressed
from .constants import METADATA_SIZE, FLOAT_SIZE


class SVDCompressor(ABC):

    @abstractmethod
    def _compress_channel(self, channel: np.ndarray, k: int) -> SVD:
        ...

    def _compress(self, img_array: np.ndarray, k: int) -> tuple[SVD, ...]:
        return tuple(self._compress_channel(img_array[..., i], k) for i in range(3))

    def to_svd(self, file_path: str, ratio: float) -> ImgCompressed:
        img_size = os.path.getsize(file_path)

        img = Image.open(file_path)
        height = img.height
        width = img.width
        k = math.floor(((img_size / ratio) - METADATA_SIZE) / (FLOAT_SIZE * 3 * (height + width + 1)))

        img_arrays = np.asarray(img)
        r, g, b = self._compress(img_arrays, k)

        return ImgCompressed(height, width, k, r, g, b)

    @staticmethod
    def from_svd(img: ImgCompressed) -> Image:
        unpacked_arrays = [img.red.to_matrix(), img.green.to_matrix(), img.blue.to_matrix()]
        image_matrix = np.dstack(unpacked_arrays).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_matrix)
