import sys

sys.path.append("../")

import config
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectModel(nn.Module):

    def __init__(self, num_rels):
        super(ObjectModel, self).__init__()
        # 两个线性层 bert特征大小->关系个数
        # 输入是一个token， (1,bert_feature_size)->(1,num_rels) 得到对于每一种关系，其是宾语的头或者尾的概率
        self.projection_heads = nn.Linear(config.bert_feature_size, num_rels).to(config.device)
        self.projection_tails = nn.Linear(config.bert_feature_size, num_rels).to(config.device)
        # 关系的个数
        self.num_rels = num_rels

    # 输入，bert的输出，主语头batch，主语尾batch
    def forward(self, bert_outputs, sub_head_batch, sub_tail_batch):
        # 我的调试

        # 增加一个模块
        sub_head_batch = sub_head_batch.unsqueeze(-1)
        sub_tail_batch = sub_tail_batch.unsqueeze(-1)
        # print(1)
        # batch索引，是一个torch列表，0-长度
        batch_idx = torch.arange(0, list(sub_head_batch.size())[0]).unsqueeze(-1).to(config.device)

        print(batch_idx.size())
        print(bert_outputs.size())
        print(sub_head_batch.size())

        # print(torch.cat([batch_idx, sub_head_batch], dim=1))
        # torch.cat将其拼接，dim=1，在列上扩展
        sub_head_feature = gather_nd(bert_outputs, torch.cat([batch_idx, sub_head_batch], dim=1))
        sub_tail_feature = gather_nd(bert_outputs, torch.cat([batch_idx, sub_tail_batch], dim=1))
        # 主语特征是主语中所有词相加然后取平均
        sub_feature = torch.mean(torch.stack([sub_head_feature, sub_tail_feature], dim=1), dim=1)

        # b，batch_size?,序列长度，特征维度
        b, seq_len, feature_dim = list(bert_outputs.size())

        # token的特征=主语特征和bert特征相加
        # 需要保证主语特征和bert输出的维度相同
        # contiguous返回一个内存连续的有相同数据的tensor，如果原tensor内存连续，则返回原tensor
        # 将主语特征拓展到bert_outputs相同的维度
        tokens_feature = sub_feature.unsqueeze(1).expand(b, seq_len, feature_dim).contiguous() + bert_outputs

        # 经过全连接层
        pred_obj_heads = self.projection_heads(tokens_feature)
        pred_obj_tails = self.projection_tails(tokens_feature)
        # 经过sigmoid层，得到概率
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)

        return pred_obj_heads, pred_obj_tails


# bert_outputs ,一个序列
def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    # indice维度的最后一个参数
    ndim = indices.shape[-1]
    # 输出的形状
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    # 扁平化的指数
    flatted_indices = indices.view(-1, ndim)
    # 切片
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    # 返回params中的参数，并转换view
    return params[slices].view(*output_shape)
