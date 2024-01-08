import torch
import torch.nn as nn
import numpy as np
from utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes
import torchvision.ops as ops


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        scores, bbox_deltas, im_info, cfg_key = input

        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)
        feat_height, feat_width = scores.size(2), scores.size(3)
        
        # Generate shifts
        shift_x = torch.arange(0, feat_width, device=scores.device) * self._feat_stride
        shift_y = torch.arange(0, feat_height, device=scores.device) * self._feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shifts = torch.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), dim=1)

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.to(scores.device)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transform predicted bbox deltas
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        scores = scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)

        # Convert anchors into proposals
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        proposals = clip_boxes(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new_zeros(batch_size, post_nms_topN, 5)
        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]
            order_single = order[i]

            if pre_nms_topN > 0:
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # Apply NMS
            keep_idx_i = ops.nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i[:post_nms_topN]

            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            num_proposal = proposals_single.size(0)
            output[i, :num_proposal, 1:] = proposals_single
            output[i, :, 0] = i

        return output
