# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            for module in [inner_block_module, layer_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


# import torch
# from torch import nn
# from torch.nn import functional as F
# from mmcv.cnn import constant_init, kaiming_init
# from torch.nn.modules.batchnorm import _BatchNorm
#
#
#
# class FPN(nn.Module):
#
#     def __init__(self, in_channels, out_channel, num_in, num_out):
#         super(FPN, self).__init__()
#         self.out_channels = out_channel
#         self.num_out = num_out
#         self.num_in = num_in
#
#         # Smooth layers
#         self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
#
#         # Lateral layers
#         self.toplayer = nn.Conv2d(2048, out_channel, kernel_size=1, stride=1, padding=0)
#         self.latlayer1 = nn.Conv2d(1024, out_channel, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d(512, out_channel, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d(256, out_channel, kernel_size=1, stride=1, padding=0)
#         self.init_weights()
#
#     def _upsample_add(self, x, y):
#         '''Upsample and add two feature maps.
#         Args:
#           x: (Variable) top feature map to be upsampled.
#           y: (Variable) lateral feature map.
#         Returns:
#           (Variable) added feature map.
#         Note in PyTorch, when input size is odd, the upsampled feature map
#         with `F.upsample(..., scale_factor=2, mode='nearest')`
#         maybe not equal to the lateral feature map size.
#         e.g.
#         original input size: [N,_,15,15] ->
#         conv2d feature map size: [N,_,8,8] ->
#         upsampled feature map size: [N,_,16,16]
#         So we choose bilinear upsample which supports arbitrary output sizes.
#         '''
#         _, _, H, W = y.size()
#         return F.interpolate(x, size=(H, W), mode='bilinear') + y
#
#     def forward(self, features):
#         assert len(features) == self.num_in
#
#         p5 = self.toplayer(features[3])
#         p4 = self._upsample_add(p5, self.latlayer1(features[2]))
#         p3 = self._upsample_add(p4, self.latlayer2(features[1]))
#         p2 = self._upsample_add(p3, self.latlayer3(features[0]))
#
#         out = self.smooth3(p2)
#         return out
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                 constant_init(m, 1)
