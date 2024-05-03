import os.path
import time

from . import directory

from .compressors import numpy_compressor, power_primitive_compressor, block_power_compressor


def process_file(ratio, src_bmp, out_bmp_dir):
    # numpy

    np_compressor = numpy_compressor.NumpySVDCompressor()
    start_time = time.time()
    tmp = np_compressor.to_svd(src_bmp, ratio)
    end_time = time.time()
    new_image = np_compressor.from_svd(tmp)
    new_image.save(os.path.join(out_bmp_dir, f'numpy-{ratio}.BMP'))

    duration = end_time - start_time
    # Power
    # http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf

    power_compressor = power_primitive_compressor.PowerSVDCompressor(duration)
    tmp = power_compressor.to_svd(src_bmp, ratio)
    new_image = power_compressor.from_svd(tmp)
    new_image.save(os.path.join(out_bmp_dir, f'power-{ratio}.BMP'))

    # Block power
    # https://www.degruyter.com/document/doi/10.1515/jisys-2018-0034/html#j_jisys-2018-0034_fig_004

    blck_power_compressor = block_power_compressor.BlockPowerSVDCompressor(duration)
    tmp = blck_power_compressor.to_svd(src_bmp, ratio)
    new_image = blck_power_compressor.from_svd(tmp)
    new_image.save(os.path.join(out_bmp_dir, f'blockpower-{ratio}.BMP'))


def main():
    src, res_dir = directory.GetPaths("src_bmp", "res_bmp")
    for (src_bmp, out_bmp_dir) in zip(src, res_dir):
        process_file(5, src_bmp, out_bmp_dir)
