#负采样

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

def neg_sample(train=None):
    ts = time.time()

    def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
        pos_data = recall_items_df[recall_items_df['label'] == 1]
        neg_data = recall_items_df[recall_items_df['label'] == 0]
        
        print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data)/len(neg_data))
        
        # 分组采样函数
        def neg_sample_func(group_df):
            neg_num = len(group_df)
            sample_num = max(int(neg_num * sample_rate), 1) # 保证最少有一个
            sample_num = min(sample_num, 5) # 保证最多不超过5个，这里可以根据实际情况进行选择
            return group_df.sample(n=sample_num, replace=False)
        
        # 对用户进行负采样，保证所有用户都在采样后的数据中
        neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
        # 对文章进行负采样，保证所有文章都在采样后的数据中
        neg_data_item_sample = neg_data.groupby('article_id', group_keys=False).apply(neg_sample_func)
        
        # 将上述两种情况下的采样数据合并
        # neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
        neg_data_new = pd.concat([neg_data_user_sample,neg_data_item_sample], ignore_index=True)
        # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重

        neg_data_new = neg_data_new.sort_values(['user_id', 'pred_score']).drop_duplicates(['user_id', 'article_id'], keep='last')
        
        # 将正样本数据合并
        data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)
        
        return data_new
    train = neg_sample_recall_data(train)
    print('Negative Data Sample Finished! Cost time: {}'.format(time.time() - ts))
    return train