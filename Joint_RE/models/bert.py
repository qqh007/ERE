import sys

import torch
import torch.nn as nn
# from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig


class BERT(BertPreTrainedModel):
    # 类类型的变量，类的别名
    config_class = BertConfig
    # pertrained_model_archieve_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    # 输入一个config，生成一个bert模型
    def __init__(self, config):
        super(BERT, self).__init__(config)
        self.bert_model = BertModel(config=config)

    # 输入，input_ids，注意力掩码，token类型的id
    # 返回输出的第一维
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用了BertModel中的结果
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return outputs[0]
