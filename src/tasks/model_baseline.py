# coding=utf-8
# Copyleft 2019 project LXRT.
import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

from lxrt.fc import FCNet, GTH
from lxrt.attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
from torch.nn.utils.weight_norm import weight_norm
import torch
import random

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = args.MAX_VQA_LENGTH

class squeeze(nn.Module):
    def __init__(self):
        super(squeeze, self).__init__()
    
    def forward(self, input):
        return input.squeeze()


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.lxrt_encoder.load(args.load_lxmert)

    def forward(self, feat, pos, sent, out={}):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :param need_bias: True for train, False for test
        :param self_supp: True when epoch>pretraining_epoches
        :return: (b, num_answer) The logit of each answers.
        """
        (lang_feats, visn_feats, l_cls, v_cls), x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)
        out['logits'] = logit

        return out