import sys

sys.path.append("../")

from tqdm import tqdm
import json
import numpy as np
import torch

import config


# lm模型，主语模型，宾语模型，评估数据，id2关系，tokener，完全符合？，输出路径
# 返回准确率，召回率，f1值
def metric(lm_model, subject_model, object_model, eval_data, id2rel, tokenizer, exact_match=False, output_path=None):
    # 如果输出路径存在
    if output_path:
        # F是一个文件
        F = open(output_path, "w")
    # 顺序，主，关系，宾语
    orders = ["subject", "relation", "object"]
    # 正确个数，预测个数，金子个数？
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10

    # 迭代评估数据
    for line in tqdm(iter(eval_data)):
        Pred_triples = set(extract_items(lm_model, subject_model, object_model, tokenizer, line["text"], id2rel))

        #         print(Pred_triples)
        # 金，三元组，应该是标签
        Gold_triples = set(line["triple_list"])

        # 如果没有完全符合，那么评估为partial_match,部分匹配，否则为精确匹配
        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (
            Pred_triples, Gold_triples)

        #         print(Pred_triples_eval)

        #         print(Gold_triples_eval)

        #         input()
        # 精确的个数，预测的个数，&是什么意思
        # 正确的而且
        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        # 预测的个数
        predict_num += len(Pred_triples_eval)
        #
        gold_num += len(Gold_triples_eval)

        # 如果输出路径存在的化
        if output_path:
            # 将结果写入json
            # json.dumps，将Python 对象编码成 JSON 字符串
            result = json.dumps({"text": line["text"],
                                 "triple_list_gold": [
                                     dict(zip(orders, triple)) for triple in Gold_triples
                                 ],
                                 "triple_list_pred": [
                                     dict(zip(orders, triple)) for triple in Pred_triples
                                 ],
                                 "new": [
                                     dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                                 ],
                                 "lack": [
                                     dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                                 ]}, ensure_ascii=False, indent=4)
            # 写入json
            F.write(result + "\n")
    # 关闭文件
    if output_path:
        F.close()

    # 评估标准
    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    print(f"correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}")

    return precision, recall, f1_score


# 抽取项
# lm模型，主语模型，宾语模型，tokenizer，in的文本 id2关系，hbar=head阈值，tbar=tail阈值
def extract_items(lm_model, subject_model, object_model, tokenizer, text_in, id2rel, h_bar=0.5, t_bar=0.5):
    # cls id seq id，不懂
    cls_token_segment_id = 0
    sequence_a_segment_id = 0

    # tokens
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(text_in) + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # 注意力mask，初始化为全1
    attention_mask = [1] * len(token_ids)
    # segment_ids = cls +seq*(n-1)
    segment_ids = [cls_token_segment_id] + ([sequence_a_segment_id] * len(token_ids))[1:]

    # 如果太长了，则截断
    if len(token_ids) > config.BERT_MAX_LEN:
        token_ids = token_ids[:config.BERT_MAX_LEN]
        segment_ids = segment_ids[:config.BERT_MAX_LEN]
        attention_mask = attention_mask[:config.BERT_MAX_LEN]

    # 转化为tensor和long
    token_ids = torch.tensor([token_ids]).long().to(config.device)
    attention_mask = torch.tensor([attention_mask]).long().to(config.device)
    segment_ids = torch.tensor([segment_ids]).long().to(config.device)

    # bert的输出=，不懂
    bert_outputs = lm_model(token_ids, attention_mask, segment_ids)[0]
    # 经过主语层的概率，logits是个什么函数，不太懂
    sub_heads_logits, sub_tails_logits = subject_model(bert_outputs)
    # cpu()将变量放在cpu上，仍为tensor，numpy将其转换为numpy类型的数组
    # 为什么要取坐标0，不懂
    sub_heads, sub_tails = np.where(sub_heads_logits.cpu().numpy()[0] > h_bar)[0], \
                           np.where(sub_tails_logits.cpu().numpy()[0] > t_bar)[0]

    # 主语s，一个列表
    subjects = []
    # sub_heads，保存了所有是主语头的下标
    for sub_head in sub_heads:
        # 比主语头大的尾，下标们
        sub_tail = sub_tails[sub_tails >= sub_head]
        # 如果长度>0
        if len(sub_tail) > 0:
            # 找到第一个
            sub_tail = sub_tail[0]
            # 截断找到主语
            subject = tokens[sub_head: sub_tail]
            # subjuects中添加一个三元组，主语tokens，主语头位置，主语尾位置
            subjects.append((subject, sub_head, sub_tail))

    # 如果主语非空，也就是说找到了一个尾
    if subjects:
        triple_list = []
        # numpy.repeat(a, repeats, axis=None)=把这些数据重复了len(subjects)次
        token_ids = np.repeat(token_ids.cpu().numpy(), len(subjects), 0)
        segment_ids = np.repeat(segment_ids.cpu().numpy(), len(subjects), 0)
        attention_mask = np.repeat(attention_mask.cpu().numpy(), len(subjects), 0)
        # 转化为tensor
        token_ids = torch.tensor(token_ids).long().to(config.device)
        attention_mask = torch.tensor(attention_mask).long().to(config.device)
        segment_ids = torch.tensor(segment_ids).long().to(config.device)
        # bert的输出
        bert_outputs = lm_model(token_ids, attention_mask, segment_ids)[0]
        # 主语的输出
        sub_heads_logits, sub_tails_logits = subject_model(bert_outputs)

        # 转化一下，不懂
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))

        # 宾语头，宾语尾=
        obj_heads_logits, obj_tails_logits = object_model(bert_outputs,
                                                          torch.tensor(sub_heads).long().to(config.device).view(-1, ),
                                                          torch.tensor(sub_tails).long().to(config.device).view(-1, ))
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, subject in enumerate(subjects):
            # 主语tokens
            sub = subject[0]
            # lstrip() 方法用于截掉字符串左边的空格或指定字符。
            # 删除token左边的##，然后连接起来
            sub = "".join([i.lstrip("##") for i in sub])
            # 删除unsued1，然后连接起来
            sub = "".join(sub.split("[unused1]"))
            # 宾语头，宾语尾，大于阈值保留下来
            obj_heads, obj_tails = np.where(obj_heads_logits.cpu()[i] > h_bar), np.where(
                obj_tails_logits.cpu()[i] > t_bar)

            # 解压然后包装起来？
            for obj_head, rel_head in zip(*obj_heads):
                # 两层遍历
                for obj_tail, rel_tail in zip(*obj_tails):
                    # 遍历头<=尾，而且两者关系相等
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        # 关系
                        rel = id2rel[rel_head]
                        # 宾语
                        obj = tokens[obj_head: obj_tail]
                        # 宾语去掉##
                        obj = "".join([i.lstrip("##") for i in obj])
                        # 宾语去掉ununsed1
                        obj = "".join(obj.split("[unused1]"))
                        # 三元组中添加元素
                        triple_list.append((sub, rel, obj))
                        break
            # 创建一个空集合必须用 set()
        triple_set = set()
        # 放入set
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        # 将set转化为1个列表然后返回
        return list(triple_set)
    else:
        return []


# 部分匹配
# 不要求完全相等
def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold
