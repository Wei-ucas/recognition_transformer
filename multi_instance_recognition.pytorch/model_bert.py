"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

# from modules.transformation import TPS_SpatialTransformerNetwork
# from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
# from modules.sequence_modeling import BidirectionalLSTM
# from modules.prediction import Attention
# from modules.resnet_aster import ResNet_ASTER, ResNet_ASTER2

from modules.bert import Bert_Ocr
from modules.bert import Config
from modules.resnet import ResNet50
from modules.neck import FPN
from modules.roicrop import MultiInstanceAlign
from modules.encoder import CnnEncoder

from torch.nn import functional as f


class ModelBert(nn.Module):

    def __init__(self, cfg):
        super(ModelBert, self).__init__()
        self.cfg = cfg
        self.backbone = ResNet50(pretrained='torchvision://resnet50')
        self.neck = FPN(cfg['feature_channels'], cfg['fpn_out_channels'], 4, 1)
        self.roialign = MultiInstanceAlign(cfg['num_instances'], cfg['roi_size'], cfg['roi_feature_step'])
        self.encoder = CnnEncoder(cfg['fpn_out_channels'] * cfg['num_instances'], cfg['encoder_channels'])
        self.voc_len = cfg['voc_len']

        opt = Config()
        opt.dim = cfg.dim
        opt.dim_c = cfg.dim  # 降维减少计算量
        opt.p_dim = 8*32  # 一张图片cnn编码之后的特征序列长度
        opt.max_vocab_size = cfg.seq_len # 一张图片中最多的文字个数, +1 for EOS
        opt.len_alphabet = cfg.voc_len  # 文字的类别个数
        self.SequenceModeling = Bert_Ocr(opt)


    def feature_extract(self, input_images):
        outs = self.backbone(input_images)
        if self.neck is not None:
            outs = self.neck(outs)
        return outs

    def forward(self, input_images, input_labels, input_boxes, input_masks):
        # s = time.time()
        feature_maps = self.feature_extract(input_images)
        # print(time.time() -s)
        # s = time.time()
        rois_features = self.roialign(feature_maps, input_boxes)
        # print(time.time() -s)
        # s = time.time()
        encoder_features = self.encoder(rois_features)
        b, c, w, h = encoder_features.shape
        sq_features = encoder_features.view(b,c,-1)
        sq_features = sq_features.permute(0,2,1)

        pad_mask = input_labels
        contextual_feature = self.SequenceModeling(sq_features, None)

        prediction = contextual_feature
        loss = self.loss(prediction, input_labels, input_masks)

        return loss, prediction

    def loss(self, pred_logits, input_labels, input_mask):
        # input_labels = f.one_hot(input_labels, num_classes=self.voc_len)
        N = pred_logits.shape[0]
        input_labels = input_labels.view(-1)
        pred_logits = pred_logits.view(-1, self.voc_len)
        input_mask = input_mask.view(-1)
        loss = f.cross_entropy(pred_logits, input_labels, reduction='none')
        loss = (input_mask.cuda() * loss).sum() / N
        return loss
        # """ Transformation stage """
        # if not self.stages['Trans'] == "None":
        #     input = self.Transformation(input)
        #
        # """ Feature extraction stage """
        # visual_feature = self.FeatureExtraction(input)
        # if self.stages['Feat'] == 'AsterRes':
        #     b, c, h, w = visual_feature.shape
        #     visual_feature = visual_feature.view(b, c, -1)
        #     visual_feature = visual_feature.permute(0, 2, 1)
        # else:
        #     visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        #     visual_feature = visual_feature.squeeze(3)
        #
        # """ Sequence modeling stage """
        # if self.stages['Seq'] == 'BiLSTM':
        #     contextual_feature = self.SequenceModeling(visual_feature)
        # elif self.stages['Seq'] == 'Bert':
        #     pad_mask = text
        #     contextual_feature = self.SequenceModeling(visual_feature, pad_mask)
        # else:
        #     contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
        #
        # """ Prediction stage """
        # if self.stages['Pred'] == 'CTC':
        #     prediction = self.Prediction(contextual_feature.contiguous())
        # elif self.stages['Pred'] == 'Bert_pred':
        #     prediction = contextual_feature
        # else:
        #     prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
        #                                  batch_max_length=self.opt.batch_max_length)
        #
        # return prediction
