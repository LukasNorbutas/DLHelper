from typing import *
from pathlib import *

import os
from .get_image_size import get_image_size

def file_getter(
    directory: str) -> List[str]:
    """
    Scan directory and its sub-directories for files and return a list with paths to each of them.

    # Arguments
        directory: path to directory to scan.
    """
    files = []
    for (path, subdirs, filenames) in os.walk(directory):
        files += [os.path.join(path, file) for file in filenames]
    return files

def get_image_resolution(
    filename: str,
    dim: str = "both") -> Union[int, Tuple[int, int]]:
    """
    Get image dimensions. Used if DataRaw.init_df is executed with "get_resolution" argument.

    # Arguments
        filename: path to the image file (e.g. "/data/train/image_1.png")
        dim: dimension(s) of the image to retrieve. "height"/"width"/"both"
    """
    height, width = get_image_size(filename)
    if dim == "both":
        return height, width
    elif dim == 1:
        return height
    elif dim == 0:
        return width
