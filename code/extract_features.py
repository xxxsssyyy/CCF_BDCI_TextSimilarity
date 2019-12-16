#coding:utf-8
import numpy as np
from tqdm import tqdm
import jieba
import time
import os
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',1000)
import re
from keras.utils.np_utils import to_categorical

"""
*************************************************************************
数据读取部分
*************************************************************************
"""
def read_data(file_path, id, name):
    train_id = []
    train_title = []
    train_text = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for idx, line in enumerate(f):
            line = line.strip().split(',')
            train_id.append(line[0].replace('\'', '').replace(' ', ''))
            train_title.append(line[1])
            train_text.append('，'.join(line[2:]))
    output = pd.DataFrame(dtype=str)
    output[id] = train_id
    output[name + '_title'] = train_title
    output[name + '_content'] = train_text
    return output

#复赛数据
train_interrelation = pd.read_csv('../data/Train_Interrelation.csv', dtype=str)
Train_Achievements = read_data('../data/Train_Achievements.csv', 'Aid', 'Achievements')
Requirements = read_data('../data/Requirements.csv', 'Rid', 'Requirements')
TestPrediction = pd.read_csv('../data/TestPrediction.csv', dtype=str)
Test_Achievements = read_data('../data/Test_Achievements.csv', 'Aid', 'Achievements')

train = pd.merge(train_interrelation, Train_Achievements, on='Aid', how='left')
train = pd.merge(train, Requirements, on='Rid', how='left')

test = pd.merge(TestPrediction, Test_Achievements, on='Aid', how='left')
test = pd.merge(test, Requirements, on='Rid', how='left')

data = pd.concat([train, test])

"""
*************************************************************************
特征工程部分
*************************************************************************
"""
# 清洗文本
def clean_line(text):
    text = re.sub("[A-Za-z0-9\!\=\？\%\[\]\,\（\）\>\<:&lt;\/#\. -----\_]", "", text)
    text = text.replace('图片', '')
    text = text.replace('\xa0', '') # 删除nbsp
    # new
    r1 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)        #去除html标签
    text = re.sub(r1,'',text)
    text = text.strip()
    return text
stop_words = pd.read_table('../stop.txt', header=None)[0].tolist()
def cut_text(sentence):
    tokens = list(jieba.cut(sentence))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens
def word_match_share(row):# 公共字符长度比例
    A = cut_text(row['Requirements_content'])
    B = cut_text(row['Achievements_content'])
    q1words = {}
    q2words = {}
    for word in A:
        q1words[word] = 1
    for word in B:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
def jaccard(row): # 分词的交集与并集比例
    query = set(cut_text(row['Requirements_content']))
    title = set(cut_text(row['Achievements_content']))
    wic = query.intersection(title) #两个集合交集
    uw = query.union(title) # 两个集合并集
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))
def common_words(row): #分词交集的数量
    query = set(cut_text(row['Requirements_content']))
    title = set(cut_text(row['Achievements_content']))
    return len(set(query).intersection(set(title)))
def total_unique_words(row): #分词并集的数量
    query = set(cut_text(row['Requirements_content']))
    title = set(cut_text(row['Achievements_content']))
    return len(set(query).union(title))
def wc_diff(row): #匹配的两文本长度差
    query = cut_text(row['Requirements_content'])
    title = cut_text(row['Achievements_content'])
    return abs(len(query) - len(title))
def wc_ratio(row): #文本长度比例
    query = cut_text(row['Requirements_content'])
    title = cut_text(row['Achievements_content'])
    l1 = len(query)*1.0
    l2 = len(title)
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
def wc_diff_unique(row):#匹配文本词量差
    query = cut_text(row['Requirements_content'])
    title = cut_text(row['Achievements_content'])
    return abs(len(set(query)) - len(set(title)))
def wc_ratio_unique(row):#匹配的两文本词量比例
    query = cut_text(row['Requirements_content'])
    title = cut_text(row['Achievements_content'])
    l1 = len(set(query)) * 1.0
    l2 = len(set(title))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
def same_start_word(row): #文本首字是否相同的bool特征
    query = cut_text(row['Requirements_content'])
    title = cut_text(row['Achievements_content'])
    if not query or not title:
        return np.nan
    return int(query[0] == title[0])
def tfidf_word_match_share(row, weights=None):
    query = cut_text(row['Requirements_content'])
    title = cut_text(row['Achievements_content'])
    q1words = {}
    q2words = {}
    for word in query:
        q1words[word] = 1
    for word in title:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in                                                                   q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R
# tfidf 所需
from collections import Counter
train_qs = pd.Series(data['Requirements_content'].apply(lambda x: cut_text(x)).tolist()
                    + data['Achievements_content'].apply(lambda x: cut_text(x)).tolist())
words = [x for y in train_qs for x in y]
counts = Counter(words)
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
weights = {word: get_weight(count) for word, count in counts.items()}

for tcol in tqdm(['Achievements_title', 'Achievements_content','Requirements_title', 'Requirements_content']):
    data[tcol+"_len"] = data[tcol].apply(lambda x:len(x))

data['word_match'] = data.apply(word_match_share, axis=1) # 1
data['jaccard'] = data.apply(jaccard, axis=1, raw=True)  # 2
data['common_words'] = data.apply(common_words, axis=1, raw=True)  # 3
data['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  # 4
data['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)  # 5
data['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)  # 6
data['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)  # 7
data['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)  # 8
data['same_start_word'] = data.apply(same_start_word, axis=1, raw=True)  # 9
data['tfidf_wm'] = data.apply(lambda x: tfidf_word_match_share(x, weights), axis=1, raw=True)  # 11
data['A_length'] = data['Achievements_content'].apply(lambda x: len(cut_text(x)))  # 12
data['B_length'] = data['Requirements_content'].apply(lambda x: len(cut_text(x)))  # 13
data['query_isin_title'] = data.apply(lambda row: 1 if row['Achievements_content'] in row['Requirements_content'] else 0, axis=1)  # 14

"""
*************************************************************************
训练集、测试集拆分
*************************************************************************
"""
train = data[:train.shape[0]]
test = data[train.shape[0]:]
base_cols = ['Guid', 'Aid', 'Rid', 'Level', 'Achievements_title',
             'Achievements_content', 'Requirements_title', 'Requirements_content']
feature_cols = [col for col in data.columns if col not in base_cols]

train_achievements = train['Achievements_title'].values
train_requirements = train['Requirements_title'].values

labels = train['Level'].astype(int).values - 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)

test_achievements = test['Achievements_title'].values
test_requirements = test['Requirements_title'].values

train_features = train[feature_cols].values
test_features = test[feature_cols].values
