import sys

sys.path.append("../")

import config
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubJectModel(nn.Module):

    # 两个线性层，计算头和尾的概率
    def __init__(self):
        super(SubJectModel, self).__init__()
        # bert特征大小->1
        # 输出为1个token，输出为该token是头或尾的概率
        self.projection_heads = nn.Linear(config.bert_feature_size, 1)
        self.projection_tails = nn.Linear(config.bert_feature_size, 1)
        self.projection_heads.to(config.device)
        self.projection_tails.to(config.device)

    def forward(self, bert_outputs):
        # 输入，bert输出
        sub_heads = self.projection_heads(bert_outputs)
        sub_tails = self.projection_tails(bert_outputs)

        # 这里不懂
        b, _, _ = list(sub_heads.size())

        # 经过sigmoid函数
        sub_heads = torch.sigmoid(sub_heads).view(b, -1)
        sub_tails = torch.sigmoid(sub_tails).view(b, -1)

        # 返回两个概率
        return sub_heads, sub_tails
