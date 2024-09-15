# Standard libraries
import os
import math
import random
import pickle
import operator


# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
from torchvision import transforms

# Setup CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)

use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)



#Seed
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)