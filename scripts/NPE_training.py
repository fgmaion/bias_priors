import numpy as np
import torch
import ili
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner

import time

data_name = 'bias_NPE_log_14.yaml'
infer_name = 'bias_NPE_large_log_14_bs_v2.yaml'

# Load Training Data
all_loader = StaticNumpyLoader.from_config("configs/data/"+data_name)

# train a model to infer theta -> x. save it as ../training_results/name/posterior.pkl
runner = InferenceRunner.from_config(
    f"configs/infer/"+infer_name)

runner(loader=all_loader, seed=1234)
