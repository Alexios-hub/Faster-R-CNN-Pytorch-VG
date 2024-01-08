import torch
from torch import nn
from torch.nn import functional as F
from RoIAlignFunction import RoIAlignFunction  # Adjust this import based on your project structure

class RoIAlign(nn.Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction.apply(features, rois, self.aligned_height, self.aligned_width, self.spatial_scale)

class RoIAlignAvg(nn.Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction.apply(features, rois, self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)
        return F.avg_pool2d(x, kernel_size=2, stride=1)

class RoIAlignMax(nn.Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction.apply(features, rois, self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)
        return F.max_pool2d(x, kernel_size=2, stride=1)
