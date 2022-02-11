import warnings
import subprocess
import shlex
import numpy as np
import matplotlib.pyplot as plt

from plotradiodata.settings import *

warnings.simplefilter('ignore', UserWarning)
arcsec = 1. / 3600.  # degree
deg = np.radians(1)  # radian
