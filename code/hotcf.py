
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

warnings.filterwarnings('ignore')


save_path='../results3/'

#获取用户最后一次点击时间前后24小时的热门文章
def get_item_topk_click_(hot_articles, hot_articles_dict,click_time, past_click_articles,k):
    topk_click = []
    # lag_max = click_time - 24 * 60 * 60 * 1000
    # lag_min = click_time + 24 * 60 * 60 * 1000
        # 热度文章在用户最后一次点击时刻起，前3小时~27小时内的文章
    # 热度文章在用户最后一次点击时刻起，前3小时~27小时内的文章
    lag_max = click_time-27 * 60 * 60 * 1000
    lag_min = click_time-3 * 60 * 60 * 1000

    for article_id in hot_articles['article_id'].unique():
        if article_id in past_click_articles:
            continue
        if not (lag_max <= hot_articles_dict[article_id] <= lag_min):
            continue
        

        topk_click.append(article_id)
        if len(topk_click) == k:
            break
    return topk_click

def hot_recall(topk=10, train_click=None,test_click=None,train_past_clicks=None, test_last_click=None,articles=None):
    ts = time.time()
    # if train:
    #     all_click = train_click.copy()
    # if test:
    #     all_click = test_click.copy()
    # if train and test:
    #     trn_click = train_click.copy()
    #     tst_click = test_click.copy()
    #     all_click = pd.concat([trn_click, tst_click], ignore_index=True)
    
    # all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    # return all_click

    train_click_df = train_click.copy().drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    test_click_df = test_click.copy().drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    
    train_click_df = train_click_df.sort_values(['user_id', 'click_timestamp'])
    test_click_df = test_click_df.sort_values(['user_id', 'click_timestamp'])
    
    articles_copy = articles.copy().rename(columns={'article_id': 'click_article_id'})
    
    #为用户点击训练集添加文章信息
    train_click_df = train_click_df.merge(articles_copy, on='click_article_id', how='left')
    test_click_df = test_click_df.merge(articles_copy, on='click_article_id', how='left')

    train_last_click = train_past_clicks.groupby('user_id').agg({'click_timestamp': 'max'}).reset_index()
    #获取每个用户的最新点击时间
    train_last_click_time = train_last_click.set_index('user_id')['click_timestamp'].to_dict()
    #获取测试集每个用户的最新点击时间
    test_last_click_time = test_last_click.set_index('user_id')['click_timestamp'].to_dict()

    #计算文章点击热度
    train_hot_articles = pd.DataFrame(train_click_df['click_article_id'].value_counts().index.to_list(), columns=['article_id'])
    #添加热门文章相关信息
    train_hot_articles = train_hot_articles.merge(articles).drop(columns=['category_id', 'words_count'])
    train_hot_articles_dict = train_hot_articles.set_index('article_id')['created_at_ts'].to_dict()

    test_hot_articles = pd.DataFrame(test_click_df['click_article_id'].value_counts().index.to_list(), columns=['article_id'])
    test_hot_articles = test_hot_articles.merge(articles).drop(columns=['category_id', 'words_count'])
    test_hot_articles_dict = test_hot_articles.set_index('article_id')['created_at_ts'].to_dict()
    
    #根据每个用户的历史点击记录，推荐出一组与该用户相关的热门文章
    train_list = []
    for user_id in tqdm(train_past_clicks['user_id'].unique()):
        user = train_past_clicks.loc[train_past_clicks['user_id'] == user_id]
#         user = user[:(len(user) - 1)]
        click_time = train_last_click_time[user_id]
        past_click_articles = user['click_article_id'].values
        item_topk_click = get_item_topk_click_(train_hot_articles, train_hot_articles_dict, click_time, past_click_articles, k=topk)
        for id in item_topk_click:
            rows = [user_id, id]
            train_list.append(rows)

    hot_train_recall = pd.DataFrame(train_list, columns=['user_id', 'article_id'])
    hot_train_recall.to_csv(save_path + 'hot_train_recall.csv', index=False)

    test_list = []
    for user_id in tqdm(test_click_df['user_id'].unique()):
        user = test_click_df.loc[test_click_df['user_id'] == user_id]
        click_time = test_last_click_time[user_id]
        past_click_articles = user['click_article_id'].values
        item_topk_click = get_item_topk_click_(test_hot_articles, test_hot_articles_dict, click_time, past_click_articles, k=topk)
        for id in item_topk_click:
            rows = [user_id, id]
            test_list.append(rows)

    hot_test_recall = pd.DataFrame(test_list, columns=['user_id', 'article_id'])
    hot_test_recall.to_csv(save_path + 'hot_test_recall.csv', index=False)

    print('Hot Recall Finished! Cost time: {}'.format(time.time() - ts))
