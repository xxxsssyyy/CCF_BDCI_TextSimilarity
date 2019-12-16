#coding:utf-8
import numpy as np
from tqdm import tqdm
import jieba
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import os
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',1000)

from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback

from extract_features import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

learning_rate = 5e-5
min_learning_rate = 1e-5
config_path = '../ckpt/bert_config.json'
checkpoint_path = '../ckpt/bert_model.ckpt'
dict_path = '../ckpt/vocab.txt'

file_path = '../log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)


token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)

class data_generator:
    def __init__(self, data, batch_size=32): # batch_size：16, 32
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, Xf, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_,F, Y = [], [], [], []
            for c, i in enumerate(idxs):
                achievements = X1[i]
                requirements = X2[i]
                # t是经过编码过后得到，纯整数集合
                # t_结果为：[0]*first_len+[1]*sencond_len,后面接max_len剩余
                t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=64)
                T.append(t)
                T_.append(t_)
                F.append(Xf[i])
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(T)
                    T_ = np.array(T_)
                    F = np.array(F)
                    Y = np.array(Y)
                    yield [T, T_, F], Y
                    T, T_,F, Y = [], [],[],[]

def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    T3 = Input(shape=(len(feature_cols),))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)  # 取出[CLS]对应的向量用来做分类
    T = Concatenate()([T, T3])

    output = Dense(4, activation='softmax')(T)

    model = Model([T1, T2, T3], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率 5e-5, 3e-5, 2e-5
        metrics=['accuracy']
    )
    model.summary()
    return model


class Evaluate(Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        #第一个epoch用来warmup，第二个epoch把学习率降到最低
        
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save_weights('../model_save/bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_x2,val_xf, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            achievements = val_x1[i]
            requirements = val_x2[i]
            Tf = np.array([val_xf[i]])
            t1, t1_ = tokenizer.encode(first=achievements, second=requirements)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_, Tf])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0]+1)
            prob.append(_prob[0])

        score = 1.0 / (1 + mean_absolute_error(val_y+1, self.predict))
        acc = accuracy_score(val_y+1, self.predict)
        f1 = f1_score(val_y+1, self.predict, average='macro')
        return score, acc, f1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

def predict(data):
    prob = []
    val_x1, val_x2, val_XF = data
    for i in tqdm(range(len(val_x1))):
        achievements = val_x1[i]
        requirements = val_x2[i]
        Tf = np.array([val_XF[i]])
        t1, t1_ = tokenizer.encode(first=achievements, second=requirements)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_, Tf])
        prob.append(_prob[0])
    return prob

oof_train = np.zeros((len(train), 4), dtype=np.float32)
oof_test = np.zeros((len(test), 4), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = train_achievements[train_index]
    x2 = train_requirements[train_index]
    xf = train_features[train_index]
    y = labels_cat[train_index]

    val_x1 = train_achievements[valid_index]
    val_x2 = train_requirements[valid_index]
    val_xf = train_features[valid_index]
    val_y = labels[valid_index]
    val_cat = labels_cat[valid_index]

    train_D = data_generator([x1, x2,xf, y])
    evaluator = Evaluate([val_x1, val_x2,val_xf, val_y, val_cat], valid_index)

    model = get_model()
    model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=5, # 可微调
                        callbacks=[evaluator]
                       )
    # 参数steps_per_epoch是通过把训练样本的数量除以批次大小得出的
    model.load_weights('../model_save/bert{}.w'.format(fold))
    oof_test += predict([test_achievements, test_requirements, test_features])
    K.clear_session()

oof_test /= 5
np.savetxt('../model_save/train_bert.txt', oof_train)
np.savetxt('../model_save/test_bert.txt', oof_test)

cv_score = 1.0 / (1 + mean_absolute_error(labels+1, np.argmax(oof_train, axis=1) + 1))
print(cv_score)
test['Level'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'Level']].to_csv('../sub/bert_{}.csv'.format(cv_score), index=False)

