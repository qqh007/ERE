from transformers import BertConfig, DistilBertConfig, AlbertConfig
from utils.tokenizer import BERTTokenizer
from transformers import BertTokenizer, BertModel
from models import BERT
from transformers.modeling_bert import BertConfig
import torch


epoches = 15
batch_size = 10
lr = 1e-5


# bert最大长度，不懂
BERT_MAX_LEN = 200
# 随机种子，干嘛的，忘记了
RANDOM_SEED = 2020
# 最大句子长度
max_text_len = 100
# bert特征维度
bert_feature_size = 768

# 是否加载权重
load_weight = False

# 一些路径
dataset_path = "./datasets/"
pretrained_model_path = "./pretrained_models/"
save_weights_path = "./saved_weights/"
model_file_path = "./saved_weights/spanbert_model_24000_1591679879"

# 这个字典好像传递的是模块
# 模型类字典， 字符串：元组(类)
MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BERTTokenizer),
    'spanbert': (BertConfig, BertModel, BERTTokenizer),
    'chinese-bert': (BertConfig, BertModel, BertTokenizer)
}

# 模型地址map
MODEL_PATH_MAP = {
    'bert': 'bert-base-cased',
    'spanbert': 'spanbert-base-cased',
    'chinese-bert':'chinese-bert_chinese_wwm_pytorch'
}

# use_cuda = True
use_cuda = False
# device = torch.device("cuda:0")
device = torch.device("cpu")

# 焦点损失
focal_loss = False


