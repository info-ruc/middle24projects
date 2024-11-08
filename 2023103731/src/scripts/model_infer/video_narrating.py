'''
Input: video (movie_id, timestamp)
Output: narration
'''

from __future__ import print_function
from __future__ import division
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append("/data5/yzh/MovieUN_v2/video-paragraph")
import argparse
import json
import time
import pdb
import random
import numpy as np

import torch
import models.transformer
from models.transformer import DECODER
import readers.caption_data as dataset
import framework.run_utils
import framework.logbase
import torch.utils.data as data

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def VideoNarrating(movie_id, starttime, endtime):

  movie_id = str(movie_id)
  starttime = str(starttime) # e.g., 1000 (第1000秒)
  endtime = str(endtime) # e.g., 2000 (第2000秒)
  
  set_seeds(12345)

  data_path = {
    'model_config': '/data4/myt/MovieChat/model_infer/config/video_narrating.json',
    'ckpt': '/data5/yzh/MovieUN_v2/video-paragraph/results_2/model/roleaware.90.th',
  }
  sys.stdout = open(os.devnull, 'w')

  model_cfg = models.transformer.TransModelConfig()
  model_cfg.load(data_path['model_config'])
  _model = models.transformer.TransModel(model_cfg, _logger=None)
  sys.stdout = sys.__stdout__

  return _model.infer(movie_id, starttime, endtime, data_path['ckpt'])


