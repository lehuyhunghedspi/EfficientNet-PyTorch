from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b0')
data = torch.zeros((1, 3, img_size, img_size))
output = net(data)
assert not torch.isnan(output).any()