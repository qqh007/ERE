import sys

# sys.path是一个列表,返回上一级地址
sys.path.append("../")

import numpy as np
import re
import json
from random import choice
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import config


# 从source里面找到匹配的target的最小下标
# 比如 source: [...aaa...] target [aaa]
def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


# sent是一个字典，"text":,"triple_list"
# tuple() 函数将列表转换为元组
# 这步有必要吗
def to_tuple(sent):
    triple_list = []
    # triple就是三元组列表
    for triple in sent["triple_list"]:
        triple_list.append(tuple(triple))
    sent["triple_list"] = triple_list


# 序列padding,padding:填充字符，默认为0
# seq不知道大小，看起来应该保证了seq的长度<=BERT_MAX_LEN
def seq_padding(seq, padding=0):
    return np.concatenate([seq, [padding] * (config.BERT_MAX_LEN - len(seq))]) if len(
        seq) < config.BERT_MAX_LEN else seq


# 加载数据
# 训练集路径，验证机路径，测试集路径，关系字典路径
def load_data(train_path, dev_path, test_path, rel_dict_path):
    # json.load将已编码的 JSON 字符串解码为 Python 对象，列表，字典
    # open() 函数用于打开一个文件，创建一个 file 对象
    train_data = json.load(open(train_path))
    dev_data = json.load(open(dev_path))
    test_data = json.load(open(test_path))
    # 还能分别解析的吗，厉害
    id2rel, rel2id = json.load(open(rel_dict_path))

    # 一个字典，id:关系类型（字符串）
    # items() 函数以列表返回可遍历的(键, 值) 元组数组。
    # 好像不需要更新吧
    id2rel = {int(i): j for i, j in id2rel.items()}
    # 关系的个数
    num_rels = len(id2rel)

    # 随机顺序，[1,2,3]->[2,3,1]
    random_order = list(range(len(train_data)))
    # 适配随机种子
    np.random.seed(config.RANDOM_SEED)
    # 洗牌？不懂
    np.random.shuffle(random_order)
    # 将训练的数据打乱
    train_data = [train_data[i] for i in random_order]

    # 转换为列表？不懂
    # sent，文本，三元组字典
    for sent in train_data:
        to_tuple(sent)
    for sent in dev_data:
        to_tuple(sent)
    for sent in test_data:
        to_tuple(sent)

    # 打印长度
    print("train data len:", len(train_data))
    print("dev data len:", len(dev_data))
    print("test data len:", len(test_data))
    # 返回数据
    return train_data, dev_data, test_data, id2rel, rel2id, num_rels


# 自定义数据集
class CustomDataset(Dataset):

    # 数据，token化，关系id字典，关系个数
    def __init__(self, data, tokenizer, rel2id, num_rels):

        # 存到类的数据中
        self.data = data
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.num_rels = num_rels

        # [CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。
        # [SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。
        # [UNK]标志指的是未知字符
        # [MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么
        # BertTokenizer中的数据
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.unk_token = tokenizer.unk_token
        # 填充token的id，不知道
        self.pad_token_id = tokenizer.pad_token_id
        # 不懂
        self.sequence_a_segment_id = 0
        self.cls_token_segment_id = 0

        self.get_examples()

    # 得到例子
    def get_examples(self):

        examples = []

        # tqdm用于显示进度
        # 在data长度范围内
        for idx in tqdm(range(len(self.data))):
            # 行
            line = self.data[idx]
            # 单词之间以空格连接，太长则截断
            text = " ".join(line["text"].split()[:config.max_text_len])
            # 通过tokener，得到tokens
            # 列表的拼接
            tokens = [self.cls_token] + self.tokenizer.tokenize(text) + [self.sep_token]

            # 长度>BERT_MAX_LEN:则截断
            if len(tokens) > config.BERT_MAX_LEN:
                tokens = tokens[:config.BERT_MAX_LEN]
            # 长度
            text_len = len(tokens)

            # 字典，主语-关系和宾语
            s2ro_map = {}
            # 对于三元组 triple应该是一个三元组，或者列表
            for triple in line["triple_list"]:
                # 将主语和宾语token化
                triple = (self.tokenizer.tokenize(triple[0]), triple[1], self.tokenizer.tokenize(triple[2]))
                # 主语头索引和宾语头索引，在文本中寻找
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                # 如果主语和宾语都存在
                if sub_head_idx != -1 and obj_head_idx != -1:
                    # sub，一个元组，标记了主语的起始位置和结束位置,[]
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    # 如果主语不在map中
                    if sub not in s2ro_map:
                        # 新建一个map项
                        s2ro_map[sub] = []
                    # 字典中添加一个项，元组(宾语开始，宾语结束，关系id)
                    s2ro_map[sub].append((obj_head_idx,
                                          obj_head_idx + len(triple[2]) - 1,
                                          self.rel2id[triple[1]]))

            # 如果字典存在，也就是说字典非空
            if s2ro_map:
                # example，一个字典
                example = {}
                # id的列表，难道是BertTokenizer里面的方法？
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                # 注意力mask初始化为全1
                attention_mask = [1] * len(token_ids)
                # segmentid=，不懂
                segment_ids = [self.cls_token_segment_id] + ([self.sequence_a_segment_id] * text_len)[1:]
                # token_ids=text_len
                assert len(token_ids) == text_len, "Error !"
                # 主语头和主语为=全0
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                # 遍历key值，也就是sub，主语
                for s in s2ro_map:
                    # 把主语特定位置设为1
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                # choice() 方法返回一个列表，元组或字符串的随机项。
                # 随机找到一个主语头，主语尾
                sub_head, sub_tail = choice(list(s2ro_map.keys()))
                # 宾语头和宾语尾，(文本长度，关系个数)
                # 这里整合了所有的关系
                obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
                # 以该元组查找，如果没有，返回空
                for ro in s2ro_map.get((sub_head, sub_tail), []):
                    # 关系宾语开头，关系宾语结尾
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1

                # 填充
                token_ids = seq_padding(token_ids)
                attention_mask = seq_padding(attention_mask)
                segment_ids = seq_padding(segment_ids)
                sub_heads = seq_padding(sub_heads)
                sub_tails = seq_padding(sub_tails)
                obj_heads = seq_padding(obj_heads, np.zeros(self.num_rels))
                obj_tails = seq_padding(obj_tails, np.zeros(self.num_rels))

                # 将这些数 据放在字典example中
                example["token_ids"] = token_ids
                example["attention_mask"] = attention_mask
                example["segment_ids"] = segment_ids
                example["sub_heads"] = sub_heads
                example["sub_tails"] = sub_tails
                example["obj_heads"] = obj_heads
                example["obj_tails"] = obj_tails
                example["sub_head"] = np.array(sub_head)  # np 形式
                example["sub_tail"] = np.array(sub_tail)
                example["text"] = text
                example["tokens"] = tokens

                # examples
                examples.append(example)

        # example和数据长度
        self.examples = examples
        self.data_len = len(self.examples)

    # 当实例对象通过[] 运算符取值时，会调用它的方法__getitem__
    def __getitem__(self, index):

        # 依照下标取到expamp
        example = self.examples[index]

        token_ids = example["token_ids"]
        attention_mask = example["attention_mask"]
        segment_ids = example["segment_ids"]

        sub_heads = example["sub_heads"]
        sub_tails = example["sub_tails"]

        obj_heads = example["obj_heads"]
        obj_tails = example["obj_tails"]

        sub_head = example["sub_head"]
        sub_tail = example["sub_tail"]

        text = example["text"]
        tokens = example["tokens"]

        #         print(text)
        #         print(tokens)

        #         print(token_ids)
        #         print(attention_mask)
        #         print(sub_heads)
        #         print(sub_tails)

        #         input()

        # 将example转换成tensor的列表
        example = [
            torch.tensor(token_ids).long().to(config.device),
            torch.tensor(attention_mask).long().to(config.device),
            torch.tensor(segment_ids).long().to(config.device),
            torch.tensor(sub_heads).float().to(config.device),
            torch.tensor(sub_tails).float().to(config.device),
            torch.tensor(obj_heads).float().to(config.device),
            torch.tensor(obj_tails).float().to(config.device),
            torch.tensor(sub_head).long().to(config.device),
            torch.tensor(sub_tail).long().to(config.device),
            text,
            tokens
        ]

        return example

    # 长度
    def __len__(self):

        return self.data_len


# 自定义batch
class CustomBatch:
    # data，不知道是什么数据
    def __init__(self, data):
        # *解析元组或map中的键
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        transposed_data = list(zip(*data))

        # torch.stack，常见的拼接函数
        # 使用stack可以保留两个信息：[1. 序列] 和 [2. 张量矩阵] 信息，属于【扩张再拼接】的函数。
        # outputs = torch.stack(inputs, dim=?) → Tensor
        #  [m,n]->[k,m,n]，其中k是长度，返回值是一个张量
        self.token_ids = torch.stack(transposed_data[0], 0)
        self.attention_mask = torch.stack(transposed_data[1], 0)
        self.segment_ids = torch.stack(transposed_data[2], 0)
        self.sub_heads = torch.stack(transposed_data[3], 0)
        self.sub_tails = torch.stack(transposed_data[4], 0)
        self.obj_heads = torch.stack(transposed_data[5], 0)
        self.obj_tails = torch.stack(transposed_data[6], 0)
        self.sub_head = torch.stack(transposed_data[7], 0)
        self.sub_tail = torch.stack(transposed_data[8], 0)
        self.text = transposed_data[9]
        self.tokens = transposed_data[10]


# 整理包装
# 返回了自定义batch，然后可以调用其参数
# 其实就是把列表表现出来的数据变成了多维德向量
def collate_wrapper(batch):
    return CustomBatch(batch)
