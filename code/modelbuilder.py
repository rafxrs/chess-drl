import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import config

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class RLModel(nn.Module):
    def __init__(self, input_shape, policy_size, value_size, num_blocks, num_filters):
        super(RLModel, self).__init__()
        # Extract proper input shape for PyTorch (channels should be first dimension)
        if len(input_shape) == 3:
            c = input_shape[2]  # Last dimension is channels
            h, w = input_shape[0], input_shape[1]  # First two dimensions are height and width
        else:
            c = input_shape[0]  # First dimension is channels
            h, w = input_shape[1], input_shape[2]  # Next dimensions are height and width
            
        self.conv = nn.Conv2d(c, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * h * w, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * h * w, 256)
        self.value_fc2 = nn.Linear(256, value_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.res_blocks(x)

        # Policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = torch.sigmoid(p)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = self.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)

        return p, v

class RLModelBuilder:
    def __init__(self, input_shape, policy_size, value_size):
        self.input_shape = input_shape
        self.policy_size = policy_size
        self.value_size = value_size
        self.num_blocks = config.AMOUNT_OF_RESIDUAL_BLOCKS
        self.num_filters = config.CONVOLUTION_FILTERS

    def build_model(self, model_path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = RLModel(self.input_shape, self.policy_size, self.value_size, 
                        self.num_blocks, self.num_filters)
        
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        return model.to(device)

# Example usage:
# builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1])
# model = builder.build_model()