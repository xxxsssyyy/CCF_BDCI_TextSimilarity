#3coding:utf-8
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',1000)
import re
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback

config_path = '../ckpt/bert_config.json'
checkpoint_path = '../ckpt/bert_model.ckpt'
dict_path = '../ckpt/vocab.txt'

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
def model():
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))


    T = bert_model([T1, T2])

    T3 = Input(shape=(100,))
    #T3 = Input(shape=(None,))
    T = Lambda(lambda x: x[:, 0])(T)  # 取出[CLS]对应的向量用来做分类

    T = Concatenate()([T,T3])

    output = Dense(4, activation='softmax')(T)

    model = Model([T1, T2, T3], output)
    model.summary()

#model()

# 将bert模型中的"字"进行编码

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
t1 = "大型石灰窑炉供风系统"
t,t2 = tokenizer.encode(first=t1)
t = np.array([t])
print(t)