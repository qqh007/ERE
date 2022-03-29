import sys
sys.path.append("../")

import numpy as np

import unicodedata
import codecs
from tqdm import tqdm
import json
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
import config

class BERTTokenizer(BertTokenizer):
    
	# 返回split_tokens
	# 输入，文本
	# 输出，列表，最后有一个unused1
    def _tokenize(self, text):
        
#         split_tokens = []
#         if self.do_basic_tokenize:
#             for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

#                 # If the token is part of the never_split set
#                 if token in self.basic_tokenizer.never_split:
#                     split_tokens.append(token)
#                 else:
#                     split_tokens += self.wordpiece_tokenizer.tokenize(token)
#         else:
#             split_tokens = self.wordpiece_tokenizer.tokenize(text)
#             split_tokens.append("[unused1]")

		# ord返回对应的 ASCII 数值
        spaced = ''
		# 跳过一些字符，空字符，未知字符，控制字符
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
		# 
        split_tokens = []
		# strip，删除首尾的空格
        for word in spaced.strip().split():
			# 使用了bert中的word_tokenizer，还没细看
            split_tokens += self.wordpiece_tokenizer.tokenize(word)
            split_tokens.append('[unused1]')
            
        return split_tokens
    
	# 输出是否为，除 \t、\r 和 \n 以外的控制字符
    def _is_control(self, ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')