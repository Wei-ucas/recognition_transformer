from modules.layers.roi_align import ROIAlign
import torch
from torch import nn
import math

class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (Tensor)
        """
        # Compute level ids
        # s = torch.sqrt(torch.cat([boxlist.area() for boxlist in boxlists]))
        s = torch.sqrt((boxlists[:,4] - boxlists[:,2]) * (boxlists[:,3] - boxlists[:,1]))
        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class MultiInstanceAlign(nn.Module):

    def __init__(self, num_instances, roi_size, strides=[4,8,16,32]):
        '''

        :param num_instances:
        :param roi_size:
        :param steps:
        '''
        super(MultiInstanceAlign, self).__init__()
        self.num_instances = num_instances
        self.roi_size = roi_size
        # self.roialign = ROIAlign(roi_size, 1.0/step, 2)
        roialigns = []
        for stride in strides:
            roialigns.append(
                ROIAlign(roi_size, 1/stride, 2)
            )
        self.roialigns = nn.ModuleList(roialigns)
        lvl_min = math.log2(strides[0])
        lvl_max = math.log2(strides[-1])
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def instance_mask(self, rois, bounding_rois):
        '''

        :param rois: N * num_instances * 5
        :param bounding_rois:  N * 5
        :return:  N * num_instances * 4
        '''
        brois_w = bounding_rois[:,3] - bounding_rois[:,1]
        brois_h = bounding_rois[:,4] - bounding_rois[:,2]
        h_ratio = self.roi_size[0] / brois_h
        w_ratio = self.roi_size[1] / brois_w
        ratios = torch.stack((w_ratio, h_ratio, w_ratio, h_ratio), dim=1).reshape(-1, 1, 4)

        bounding_rois = bounding_rois.reshape(-1,1,5)
        bounding_lt = bounding_rois[:,:,1:3].repeat(1,1,2)
        ins_pos = rois[:,:,1:] - bounding_lt # N * num_instances * 4
        ins_pos = ins_pos * ratios
        ins_pos = ins_pos.long()
        ins_pos = ins_pos.permute(1, 0, 2)# num_instances * N *4

        ins_mask = []
        N = bounding_rois.shape[0]
        for i in range(self.num_instances):
            tmp_mask = torch.zeros((N, 1, *self.roi_size), dtype=torch.float32, requires_grad=False).cuda() #N *1* W * H
            for j in range(bounding_rois.shape[0]):
                tmp_mask[j, :, ins_pos[i,j, 1]:ins_pos[i,j, 3],ins_pos[i,j, 0]:ins_pos[i,j, 2] ] = \
                    1.0
            ins_mask.append(tmp_mask)



        return ins_mask



    def forward(self, feature_maps, rois):
        '''

        :param feature_maps: Nb * C * H * W
        :param rois: N * num_instances * 5 (N = SUM(ni)) , batch_id min_x, min_y, max_x, max_y
        :return:
        '''
        batch_ids = rois[:, 0, 0]
        min_x = rois[:, :, 1].min(dim=1)[0]
        min_y = rois[:, :, 2].min(dim=1)[0]
        max_x = rois[:, :, 3].max(dim=1)[0]
        max_y = rois[:, :, 4].max(dim=1)[0]
        min_bounding_bbox = torch.stack((batch_ids, min_x, min_y, max_x, max_y),dim=1)

        num_rois = len(batch_ids)
        num_levels = len(self.roialigns)
        num_channels = feature_maps[0].shape[1]
        output_h = self.roi_size[0]
        output_w = self.roi_size[1]
        # assert num_levels == len(feature_maps)


        if num_levels == 1:
            rois_features = self.roialigns[0](feature_maps, min_bounding_bbox) # N * C * h * w
        else:
            assert num_levels == len(feature_maps)
            levels = self.map_levels(min_bounding_bbox)
            rois_features = torch.zeros(
                (num_rois, num_channels, output_h, output_w),
                dtype=feature_maps[0].dtype,
                device=feature_maps[0].device,
            )
            for level, (per_level_feature, pooler) in enumerate(zip(feature_maps, self.roialigns)):
                idx_in_level = torch.nonzero(levels==level).squeeze(1)
                rois_per_level = min_bounding_bbox[idx_in_level]
                rois_features[idx_in_level] = pooler(per_level_feature, rois_per_level)

        ins_mask = self.instance_mask(rois, min_bounding_bbox) # N * num_instances * 4

        # ins_pos = ins_pos.permute(1,0,2) # num_instances * N *4
        # ins_features = []
        # for i in range(self.num_instances):
        #     tmp_features = torch.zeros_like(rois_features, requires_grad=True)
        #     for j in range(rois_features.shape[0]):
        #         tmp_features[j, :, ins_pos[i,j, 1]:ins_pos[i,j, 3],ins_pos[i,j, 0]:ins_pos[i,j, 2] ] = \
        #             rois_features[j, :, ins_pos[i,j, 1]:ins_pos[i,j, 3],ins_pos[i,j, 0]:ins_pos[i,j, 2] ]
        #     ins_features.append(tmp_features)
        ins_features = []
        for i in range(self.num_instances):
            ins_features.append(rois_features* ins_mask[i].cuda())
            # break

        cat_features = torch.cat(ins_features, dim=1) # N * (num_instances * C) * h * w
        return cat_features
        # return rois_features


