import os
import json
# 参数解析
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_loader import load_data, collate_wrapper, CustomDataset, CustomBatch
from utils.metric import metric
import config
from models import SubJectModel, ObjectModel

# 设置可见的cuda设备，这里使用0号设备和1号设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class Trainer:

    def __init__(self, dataset):

        # train_path = config.dataset_path + dataset + "/train_data_50000.json"
        # dev_path = config.dataset_path + dataset + "/dev_data_5000.json"
        # test_path = config.dataset_path + dataset + "/dev_data_5000.json"
        # rel_dict_path = config.dataset_path + dataset + "/rel2id.json"
        # 路径
        train_path = config.dataset_path + dataset + "/train_triples.json"
        dev_path = config.dataset_path + dataset + "/dev_triples.json"
        test_path = config.dataset_path + dataset + "/test_triples.json"
        rel_dict_path = config.dataset_path + dataset + "/rel2id.json"

        # data process
        # 训练集，测试集，验证集，
        # 字典，id和relation
        # int,rel个数
        self.train_data, self.dev_data, self.test_data, self.id2rel, self.rel2id, self.num_rels = load_data(train_path,
                                                                                                            dev_path,
                                                                                                            test_path,
                                                                                                            rel_dict_path)

    # 初始化
    # 输入mode 1个字符串，表示使用哪个模型
    def setup(self, model):
        # bert的配置地址
        bert_config_path = config.pretrained_model_path + config.MODEL_PATH_MAP[model] + "/config.json"
        bert_model_path = config.pretrained_model_path + config.MODEL_PATH_MAP[model] + "/model.bin"
        bert_vocab_path = config.pretrained_model_path + config.MODEL_PATH_MAP[model] + "/vocab.txt"

        # 语言模型，这里用的是bert
        lm_config = config.MODEL_CLASSES[model][0].from_pretrained(bert_config_path)
        self.lm_model = nn.DataParallel(config.MODEL_CLASSES[model][1].from_pretrained(bert_model_path, config=lm_config)).to(config.device)
        # self.lm_tokenizer = config.MODEL_CLASSES[model][2](bert_vocab_path, do_lower_case=False)
        # tokenizer
        self.lm_tokenizer = config.MODEL_CLASSES[model][2](bert_vocab_path)
        # 训练数据
        self.train_data = CustomDataset(self.train_data,
                                        self.lm_tokenizer,
                                        self.rel2id,
                                        self.num_rels)

        # set data loader
        # pytorch中的data loader
        # collate_fn整理包装
        self.train_batcher = DataLoader(self.train_data,
                                        config.batch_size,
                                        drop_last=True,
                                        shuffle=True,
                                        collate_fn=collate_wrapper)

        # 模型
        # 使用nn.DataParallel函数来用多个GPU来加速训练。
        self.subject_model = nn.DataParallel(SubJectModel()).to(config.device)
        self.object_model = nn.DataParallel(ObjectModel(self.num_rels)).to(config.device)
        # 交叉熵损失
        self.criterion = nn.BCELoss(reduction="none")
        # 模型参数，由3个模型的参数拼接而成
        self.models_params = list(self.lm_model.parameters()) + list(self.subject_model.parameters()) + list(self.object_model.parameters())
        # 优化器
        self.optimizer = torch.optim.Adam(self.models_params, lr=config.lr)

        self.start_step = None

        # 如果加载了数据，说明是测试
        if config.load_weight:
            print("start loading weight...")

            state = torch.load(config.model_file_path, map_location=lambda storage, location: storage)
            self.lm_model.module.load_state_dict(state['lm_model'])
            self.object_model.module.load_state_dict(state['object_model'])
            self.subject_model.module.load_state_dict(state['subject_model'])

            self.start_step = state['step']

            self.optimizer.load_state_dict(state['optimizer'])
            # 如果用了cuda，就怎么样
            if config.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    # 保存模型
    def save_models(self, args, total_loss, step):
        # 字典保存参数
        state = {
            "step": step,
            "lm_model": self.lm_model.module.state_dict(),
            "object_model": self.object_model.module.state_dict(),
            "subject_model": self.subject_model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_loss": total_loss
        }
        # 保存路径
        model_save_path = os.path.join(config.save_weights_path,
                                       "%s_model_%d_%d" % (args.model, step, int(time.time())))
        # torch保存
        torch.save(state, model_save_path)

    # 训练迭代器
    def trainIters(self, args):
        # 打开train模式
        self.lm_model.train()
        self.subject_model.train()
        self.object_model.train()

        # step，如果有step的话，那么就使用已经加载的step
        if self.start_step:
            step = self.start_step
        else:
            step = 0

        # 开始训练
        for epoch in range(config.epoches):

            for batch in self.train_batcher:
                total_loss, sub_entities_loss, obj_entities_loss = self.train_one_batch(batch)
                print("epoch:", epoch, " step: ", step, "total_loss:", total_loss, "sub_entities_loss:",sub_entities_loss, "obj_entities_loss: ", obj_entities_loss)
                step += 1
                # 跑了500次之后，输出一下prec,recall f1
                if step % 500 == 0:
                    with torch.no_grad():
                        # eval模式
                        self.lm_model.eval()
                        self.subject_model.eval()
                        self.object_model.eval()

                        precision, recall, f1 = metric(self.lm_model, self.subject_model, self.object_model,
                                                       self.test_data, self.id2rel, self.lm_tokenizer,
                                                       output_path="./result.json")
                        print("precision: ", precision, "recall: ", recall, "f1: ", f1)
                        # self.save_models(args, total_loss, step)

                        # 打开一个文件写入内容
                        f = open("result3.txt", 'a+')
                        f.write("epoch: " + str(epoch) + " step: " + str(step) + " precision: " + str(precision)
                                + " recall: " + str(recall) + " f1: " + str(f1) + '\n')
                        f.close()
                    # train模式
                    self.lm_model.train()
                    self.subject_model.train()
                    self.object_model.train()
            # 重新设置train_batcher，重新申请了一个
            self.reset_train_dataloader()

    # 重新设置训练dataloader
    def reset_train_dataloader(self):
        self.train_batcher = DataLoader(self.train_data,
                                        config.batch_size,
                                        drop_last=True,
                                        shuffle=True,
                                        collate_fn=collate_wrapper)

    # 训练一个batch
    def train_one_batch(self, batch):

        # 优化器，梯度清零
        self.optimizer.zero_grad()

        # token 注意力层 segment层
        tokens_batch = batch.token_ids
        attention_mask_batch = batch.attention_mask
        segments_batch = batch.segment_ids

        # 主语头尾，head和heads有什么区别？
        # 训练集的数据，tensor形式
        # [10,200]
        sub_heads_batch = batch.sub_heads
        sub_tails_batch = batch.sub_tails
        # 训练集的数据,np array形式
        # [10,]
        sub_head_batch = batch.sub_head
        sub_tail_batch = batch.sub_tail

        # 宾语头尾
        # 训练集的数据
        # [10,200,num_rel]
        obj_heads_batch = batch.obj_heads
        obj_tails_batch = batch.obj_tails

        # 字典，bert的输入
        bert_inputs = {"input_ids": tokens_batch,
                       "attention_mask": attention_mask_batch,
                       "token_type_ids": segments_batch}
        # bert的输出，**bert将字典解析为参数
        bert_outputs = self.lm_model(**bert_inputs)[0]

        # 经过主语层，主语层的预测结果
        sub_heads, sub_tails = self.subject_model(bert_outputs)
        # 经过宾语层，宾语层的输入使用了真实标签
        pred_obj_heads, pred_obj_tails = self.object_model(bert_outputs,
                                                           sub_head_batch,
                                                           sub_tail_batch)

        # 主语损失
        sub_heads_loss = self.criterion(sub_heads, sub_heads_batch)
        sub_tails_loss = self.criterion(sub_tails, sub_tails_batch)
        # 宾语损失
        obj_heads_loss = self.criterion(pred_obj_heads, obj_heads_batch)
        obj_tails_loss = self.criterion(pred_obj_tails, obj_tails_batch)

        # 如果焦点函数
        # 采取了不同的损失计算方式
        # 这里也算是一个注意力机制
        if config.focal_loss:

            sub_entities_loss = ((torch.abs(sub_heads_batch - sub_heads) * sub_heads_loss +
                                  torch.abs(
                                      sub_tails_batch - sub_tails) * sub_tails_loss) * attention_mask_batch).sum() / attention_mask_batch.sum()

            obj_entities_loss = ((torch.abs(obj_heads_batch - pred_obj_heads) * obj_heads_loss +
                                  torch.abs(
                                      obj_tails_batch - pred_obj_tails) * obj_tails_loss) * attention_mask_batch.unsqueeze(
                -1)).sum() / attention_mask_batch.unsqueeze(-1).sum()
        else:
            # 这里使用了注意力机制，对于每一个点，加权平均
            # *，矩阵乘法
            sub_entities_loss = ((sub_heads_loss + sub_tails_loss) * attention_mask_batch).sum() / attention_mask_batch.sum()
            # 这里还可以再看一下
            obj_entities_loss = ((obj_heads_loss + obj_tails_loss) * attention_mask_batch.unsqueeze(
                -1)).sum() / attention_mask_batch.unsqueeze(-1).sum()

        # 总体损失=主语损失+宾语损失
        total_loss = sub_entities_loss + obj_entities_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

        # 返回，总体损失，主语实体的损失，宾语实体损失
        # 一个元组
        return total_loss.item(), sub_entities_loss.item(), obj_entities_loss.item()


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description="Train scrip")
    parser.add_argument("--model", default="bert", type=str, help="specify the type of language models")
    parser.add_argument("--dataset", default="NYT", type=str, help="specify the dataset")
    args = parser.parse_args()
    # 训练器
    trainer = Trainer(args.dataset)
    trainer.setup(args.model)
    trainer.trainIters(args)
