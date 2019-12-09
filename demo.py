from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
net = EfficientNet.from_name('efficientnet-b0')
img_size=512
data = torch.zeros((1, 3, img_size, img_size))
output = net(data)
assert not torch.isnan(output).any()