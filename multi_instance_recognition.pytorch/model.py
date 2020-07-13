import torch
from torch import nn
from torch.nn import functional as f
from modules.resnet import ResNet50
from modules.neck import FPN, LastLevelMaxPool
from modules.roicrop import MultiInstanceAlign
from modules.encoder import CnnEncoder
from modules.decoder import Decoder
from data_tools.data_utils import get_vocabulary
import time


class MultiInstanceRecognition(nn.Module):

    def __init__(self,cfg):

        super(MultiInstanceRecognition, self).__init__()
        self.cfg = cfg
        self.backbone = ResNet50(pretrained='torchvision://resnet50')
        self.neck = FPN(cfg['feature_channels'], cfg['fpn_out_channels'], LastLevelMaxPool())
        self.roialign = MultiInstanceAlign(cfg['num_instances'], cfg['roi_size'], cfg['roi_feature_steps'])
        self.encoder = CnnEncoder(cfg['fpn_out_channels']*cfg['num_instances'],cfg['encoder_channels'])
        self.voc_len = cfg['voc_len']
        self.decoder = Decoder(self.cfg)
        self.num_instances = cfg['num_instances']

    def feature_extract(self, input_images):
        outs = self.backbone(input_images)
        if self.neck is not None:
            outs = self.neck(outs)
        return outs

    def forward(self, input_images, input_labels, input_boxes, input_masks):
        # input_labels = input_labels.view(-1,25)
        # input_boxes = input_boxes.view(-1,self.num_instances, 5)
        # input_masks = input_masks.view(-1, 25)
        # s = time.time()


        # rois = self.roialign(input_images, input_boxes)
        # rois = rois[:,0:3,:,:]
        feature_maps = self.feature_extract(input_images)

        # print(time.time() -s)
        # s = time.time()

        rois_features = self.roialign(feature_maps, input_boxes)
        del feature_maps
        # rois_features = self.feature_extract(rois)
        # rois_features = rois_features[0].repeat(1,4,1,1)

        # print(time.time() -s)
        # s = time.time()
        encoder_features = self.encoder(rois_features)
        del rois_features
        # print(time.time() -s)
        # s = time.time()
        decoder_logits = self.decoder(encoder_features, input_labels)
        # print(time.time() - s)
        # s = time.time()

        loss = self.loss(decoder_logits, input_labels, input_masks)
        # print(time.time() - s)
        # print('-----')
        return loss, decoder_logits


    def loss(self, pred_logits, input_labels, input_mask):
        # input_labels = f.one_hot(input_labels, num_classes=self.voc_len)
        N = pred_logits.shape[0]
        input_labels = input_labels.view(-1)
        pred_logits = pred_logits.view(-1, self.voc_len)
        input_mask = input_mask.view(-1)
        loss = f.cross_entropy(pred_logits, input_labels,  reduction='none')
        loss = (input_mask.cuda() * loss).sum() / N
        return loss
#
# class Config(object):
#     '''参数设置'''
#     dim = 512
#     seq_len = 25
#     voc_len = get_vocabulary("ALLCASES_SYMBOLS")
#     '''label attentionparams'''
#     embedding = dict(
#         dim=dim,
#         embedding_dim=512,
#         pos_dim=seq_len,
#         drop_rate=0.1,
#     )
#
#     '''label att params'''
#     label_att = dict(
#         n_heads=4,
#         indim=embedding["embedding_dim"],
#         outdim=dim,
#         drop_rate=0.1
#     )
#
#     '''img label att params'''
#     att = dict(
#         n_heads=4,
#         indim=1024,
#         outdim=dim,
#         drop_rate=0.1
#     )
#
#     ff = dict(
#         dim_ff = dim * 2,
#         dim=dim
#     )


    # p_drop_attn = 0.1
    # p_drop_hidden = 0.1
    # dim = 512  # the encode output feature
    # attention_layers = 2  # the layers of transformer
    # n_heads = 8
    # dim_ff = 1024   # 位置前向传播的隐含层维度
    #
    #
    # dim_c = dim
    # seq_len = 25  # 一张图片含有字符的最大长度
