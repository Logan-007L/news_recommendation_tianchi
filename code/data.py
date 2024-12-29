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

from itemcf import itemcf_recall
from hotcf import hot_recall
from negative_sample import neg_sample
from recall import get_train_recall,get_test_recall
warnings.filterwarnings('ignore')
ts = time.time()
data_path = '../data/' # 天池平台路径
save_path = '../results3/'  # 天池平台路径

def get_past_click(train_click, test_click):
    train = train_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
    list1 = []
    train_indexs = []

    print('获取用户最后一次点击记录')
    for user_id in tqdm(train['user_id'].unique()):
        user = train[train['user_id'] == user_id]
        #表示用户最后一次点击的记录
        row = user.tail(1)
        #最后点击的索引
        train_indexs.append(row.index.values[0])
        #testA中有一些只点了一次的用户要去掉
        if len(user) >= 2:
            list1.append(row.values.tolist()[0])
    train_last_click = pd.DataFrame(list1, columns=['index', 'user_id', 'article_id', 'click_timestamp', 'click_environment',\
                                    'click_deviceGroup', 'click_os', 'click_country', 'click_region',
                                    'click_referrer_type'])
    
    #最好一次点击的数据，做为验证集
    train_last_click = train_last_click.drop(columns=['index'])

    #除了最后一次点击的数据，做为训练集
    train_past_clicks = train[~train.index.isin(train_indexs)]
    train_past_clicks = train_past_clicks.drop(columns=['index'])
    
    test = test_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
    list2 = []
    print('测试集获取用户最后一次点击记录')
    for user_id in tqdm(test['user_id'].unique()):
        user = test[test['user_id'] == user_id]
        row = user.tail(1)
        list2.append(row.values.tolist()[0])
    test_last_click = pd.DataFrame(list2, columns=['index', 'user_id', 'article_id', 'click_timestamp', 'click_environment',\
                                    'click_deviceGroup', 'click_os', 'click_country', 'click_region',
                                    'click_referrer_type'])
    test_last_click = test_last_click.drop(columns=['index'])
    
    ###                    注释要去掉↓
    all_click_df = pd.concat([train_past_clicks,test_click],ignore_index=True)
    all_click_df = all_click_df.reset_index().drop(columns=['index'])

    all_click_df = all_click_df.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

    #保存结果
    all_click_df.to_csv(save_path+'all_click_df.csv', index=False)
    train_past_clicks.to_csv(save_path+'train_past_clicks.csv', index=False)
    train_last_click.to_csv(save_path+'train_last_click.csv',index=False)
    test_last_click.to_csv(save_path+'test_last_click.csv',index=False)
    # return all_click_df, train_past_clicks, train_last_click, test_last_click
