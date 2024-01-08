import torch
from torch.autograd.function import Function

# Assuming roi_align is an appropriate module in PyTorch 2.x or a custom implementation
from . import roi_align

class RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, aligned_height, aligned_width, spatial_scale):
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        ctx.aligned_height = aligned_height
        ctx.aligned_width = aligned_width
        ctx.spatial_scale = spatial_scale

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, aligned_height, aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(aligned_height, aligned_width,
                                             spatial_scale, features,
                                             rois, output)
        else:
            roi_align.roi_align_forward(aligned_height, aligned_width,
                                        spatial_scale, features,
                                        rois, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        aligned_height, aligned_width, spatial_scale = ctx.aligned_height, ctx.aligned_width, ctx.spatial_scale

        grad_input = rois.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_align.roi_align_backward_cuda(aligned_height, aligned_width,
                                          spatial_scale, grad_output,
                                          rois, grad_input)
        return grad_input, None, None, None, None

# Use the function as follows:
# output = RoIAlignFunction.apply(features, rois, aligned_height, aligned_width, spatial_scale)
