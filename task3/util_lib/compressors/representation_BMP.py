import struct

import numpy as np
from .constants import FLOAT_SIZE, METADATA_SIZE


class SVD:
    def __init__(self, u: np.ndarray, s: np.ndarray, vt: np.ndarray):
        self.u = u
        self.s = s
        self.vt = vt

    def to_bytes(self) -> bytes:
        return self.u.astype(np.float32).tobytes() + \
            self.s.astype(np.float32).tobytes() + \
            self.vt.astype(np.float32).tobytes()

    @staticmethod
    def from_bytes(data: bytes, height: int, width: int, k: int):
        float_array = [struct.unpack('<f', data[i: i + FLOAT_SIZE])
                       for i in range(0, len(data), FLOAT_SIZE)]
        float_np_array = np.array(float_array)

        u = float_np_array[: height * k].reshape(height, k)
        s = float_np_array[height * k: (height + 1) * k].ravel()
        vh = float_np_array[(height + 1) * k:].reshape(k, width)

        return SVD(u, s, vh)

    def to_matrix(self) -> np.array:
        return self.u @ np.diag(self.s) @ self.vt


class ImgCompressed:
    def __init__(self, height, width, k: int, r: SVD, g: SVD, b: SVD):
        self.height = height
        self.width = width
        self.k = k
        self.red = r
        self.green = g
        self.blue = b

    def to_bytes(self) -> bytes:
        return struct.pack('<h', self.height) + \
            struct.pack('<h', self.width) + \
            self.red.to_bytes() + self.blue.to_bytes() + self.green.to_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        size = data.__len__()
        height = struct.unpack('<h', data[0: 2])[0]
        width = struct.unpack('<h', data[2: 4])[0]
        k = (size - METADATA_SIZE) // (FLOAT_SIZE * 3 * (height + 1 + width))
        red = SVD.from_bytes(data[METADATA_SIZE: METADATA_SIZE + ((height + width + 1) * k * FLOAT_SIZE)],
                             height, width, k)
        blue = SVD.from_bytes(data[METADATA_SIZE + ((height + width + 1) * k * FLOAT_SIZE):
                                   METADATA_SIZE + ((height + width + 1) * k * FLOAT_SIZE * 2)], height, width, k)
        green = SVD.from_bytes(data[METADATA_SIZE + ((height + width + 1) * k * FLOAT_SIZE * 2):],
                               height, width, k)
        return ImgCompressed(height, width, k, red, green, blue)
