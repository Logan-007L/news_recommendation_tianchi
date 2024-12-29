import collections
import gc
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from operator import itemgetter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')


save_path='../results3/'

def get_test_recall(itemcf=False, hot=False):
    test_recall = pd.DataFrame()
    if itemcf:
        itemcf_test_recall = pd.read_csv(save_path + 'itemcf_test_recall.csv')
        itemcf_test_recall = itemcf_test_recall.rename(columns={'click_article_id': 'article_id'})
        test_recall = pd.concat([test_recall, itemcf_test_recall], ignore_index=True)
    if hot:
        hot_test_recall = pd.read_csv(save_path + 'hot_test_recall.csv')
        test_recall = pd.concat([test_recall, hot_test_recall], ignore_index=True)
    
    test_recall = test_recall.drop_duplicates(['user_id', 'article_id'])
    test_recall.to_csv(save_path + 'test_recall.csv', index=False)
    print('Test Recall Finished!')
    return test_recall

#训练集召回
def get_train_recall(itemcf=False, hot=False, train_last_click=None,word2vec=False):

    train = pd.DataFrame()
    if itemcf:
        itemcf_train_recall = pd.read_csv(save_path + 'itemcf_train_recall.csv') 
        itemcf_train_recall = itemcf_train_recall.rename(columns={'click_article_id': 'article_id'})
        itemcf_train_recall = itemcf_train_recall.merge(train_last_click, on=['user_id', 'article_id'], how='left')
        #将召回结果，根据用户是否点击过，来添加标签
        itemcf_train_recall['label'] = itemcf_train_recall['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
        print('Train ItemCF RECALL:{}%'.format((itemcf_train_recall['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))
        train = pd.concat([train, itemcf_train_recall], ignore_index=True)
    if hot:
        hot_train_recall = pd.read_csv(save_path + 'hot_train_recall.csv')
        #将召回结果，根据用户是否点击过，来添加标签
        hot_train_recall['label'] = hot_train_recall.merge(train_last_click, on=['user_id', 'article_id'], how='left')['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
        print('Train Hot RECALL:{}%'.format((hot_train_recall['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))
        train = pd.concat([train, hot_train_recall], ignore_index=True)
    
    if word2vec:
        w2v_train_recall = pd.read_pickle(save_path+'recall_w2v.pkl')
        w2v_train_recall = itemcf_train_recall.merge(train_last_click, on=['user_id', 'article_id'], how='left')
        w2v_train_recall['label'] = itemcf_train_recall['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
        print('Train Hot RECALL:{}%'.format((w2v_train_recall['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))
        train = pd.concat([train, w2v_train_recall], ignore_index=True)
    #删除数据框 train 中的重复行
    train = train.drop_duplicates(['user_id', 'article_id'])

    train['pred_score'] = train['pred_score'].fillna(-100)
    train['sim_score'] = train['sim_score'].fillna(-100)
    print('Train Total RECALL:{}%'.format((train['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))
    print('Train Total Recall Finished!')
    train.to_csv(save_path + 'train_recall.csv', index=False)
    return train
