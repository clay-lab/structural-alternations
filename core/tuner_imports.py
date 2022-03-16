# tuner_imports.py
# 
# load modules used with tuner.py

import os
import re
import sys
import json
import gzip
import hydra
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import random
import logging
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import torch.nn as nn

from glob import glob
from copy import deepcopy
from tqdm import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from math import ceil, floor, sqrt, comb, perm
from typing import *
from shutil import copyfileobj
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.stats import pearsonr
from transformers import logging as lg
lg.set_verbosity_error()
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from sklearn.manifold import TSNE

from core.tuner_utils import *