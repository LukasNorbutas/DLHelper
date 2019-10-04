from typing import *

from .Augmentor import Augmentor
from .DataRaw import DataRaw
from .Resizer import Resizer

"""
Sub-module for storing (and keeping track of) custom datatypes.
"""

AugmentorType = TypeVar('Augmentor', bound=Augmentor)
DataRawType = TypeVar('DataRaw', bound=DataRaw)
ResizerType = TypeVar('Resizer', bound=Resizer)
