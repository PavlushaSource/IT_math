import os
from pathlib import Path


def GetPaths(path, resPath):
    projDirectory = os.getcwd()
    inputDirectory = [x[0] for x in os.walk(os.path.join(projDirectory, path))]
    src_bmp = []
    res_bmp_dir = []
    for srcDir in inputDirectory:
        for f in os.listdir(srcDir):
            if Path(f).suffix == '.bmp':
                src_bmp.append(os.path.join(projDirectory, srcDir, f))
                output_dir = os.path.join(projDirectory, resPath, Path(srcDir).name, Path(f).stem)
                os.makedirs(output_dir, exist_ok=True)
                res_bmp_dir.append(output_dir)
    return src_bmp, res_bmp_dir
