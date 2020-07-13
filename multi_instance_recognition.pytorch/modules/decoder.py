import torch
from torch import nn
from .transformer_modules import Embeddings, MultiHeadAttention, PositionWiseFeedForward
import time

# class Embedding(nn.Module):
#
#     def __int__(self, voc_len, seq_len, embedding_dim):
#         super(Embedding, self).__init__()
#         self.letter_embedding = nn.Embedding(voc_len, embedding_dim)
#         self.pos_embedding = nn.Embedding(seq_len, embedding_dim)
#
#     def forward(self, input_labels):
#         letter_embed = self.letter_embedding(input_labels)
#         seq_len = input_labels.size(1)
#         pos = torch.arange(seq_len, dtype=torch.long, device=input_labels.device)
#         pos = pos.unsqueeze(0).expand(input_labels.size(0), -1)  # (S,) -> (B, S)
#
#         e = letter_embed + self.pos_embedding(pos)
#         # TODO : LayerNorm, dropout
#         return e


class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.num_instances = cfg['num_instances']
        self.seq_len = cfg['seq_len']
        # self.embedding_dim = embedding_dim
        # self.att_dim = att_dim
        self.START_TOKEN = cfg['voc_len'] - 3
        self.PAD_TOKEN = cfg['voc_len'] -2


        self.label_embedding = Embeddings(self.cfg['embedding'])

        self.label_self_att = MultiHeadAttention(cfg['label_att']) #dim, p_drop_attn,n_heads
        self.img_label_att = MultiHeadAttention(cfg['att'])
        self.ff = PositionWiseFeedForward(cfg['ff'])

        self.decoder = nn.Linear(cfg['dim'], cfg['voc_len'])

    def do_att_op(self, img_features, labels_after_self_att):
        query = labels_after_self_att
        N, H, W, C = img_features.shape
        key_vec = img_features.reshape(-1, (H*W), C)
        outputs = self.img_label_att(query, key_vec, key_vec)
        outputs = self.ff(outputs)
        return outputs

    def forward(self, rois_features, input_labels_):
        '''

        :param rois_features: # N * (C*num_instances) * H * W
        :param input_labels: # N * 25
        :return:
        '''
        # s = time.time()
        input_labels = input_labels_.clone()
        N = input_labels.shape[0]
        rois_features = rois_features.permute(0,2,3,1) # N  * H * W * (C*num_instances)
        # print(time.time() - s)
        if self.training:
            # Add SOS to the sequence
            # input_labels = torch.split(input_labels,1, dim=1)[:-1]
            # s = time.time()
            input_labels[:, 1:] = input_labels[:, :-1]
            start_padding = torch.zeros(N, dtype=torch.long, device=input_labels.device).fill_(self.START_TOKEN)
            input_labels[:, 0] = start_padding

            # Embedding sequence labels
            labels_embed = self.label_embedding(input_labels) # N * T * embed_dim
            self_att_outputs = self.label_self_att(labels_embed, labels_embed, labels_embed,  causality=True) # N * T * dim
            glimps = self.do_att_op(rois_features, self_att_outputs) #  N * T * dim
            logits = self.decoder(glimps)

        else:
            predicts = (torch.zeros((N, self.seq_len),dtype=torch.long,  device=input_labels.device).fill_(self.PAD_TOKEN))
            pre_pred = torch.zeros(N,dtype=torch.long,  device=input_labels.device).fill_(self.START_TOKEN)
            for t in range(self.seq_len):
                predicts[:,t] = pre_pred
                # input_labels = torch.stack(predicts, dim=1)
                input_labels = predicts
                # Embedding sequence labels
                labels_embed = self.label_embedding(input_labels)  # N * T * embed_dim

                self_att_outputs = self.label_self_att(labels_embed, labels_embed, labels_embed, causality=True)  # N * T * dim

                glimps = self.do_att_op(rois_features, self_att_outputs)  # N * T * dim

                logits = self.decoder(glimps)

                current_logits = logits[:,t]

                pre_pred = torch.argmax(current_logits, dim=1)

        return logits
