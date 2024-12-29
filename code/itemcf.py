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

def itemcf_recall(topk=10,articles=None):
    # 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
    def get_user_item_time(click_df):
        
        click_df = click_df.sort_values('click_timestamp')
        
        def make_item_time_pair(df):
            return list(zip(df['click_article_id'], df['click_timestamp']))
        
        user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_pair(x))\
                                                                .reset_index().rename(columns={0: 'item_time_list'})
        user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
        
        return user_item_time_dict

    # 计算物品相似度
    def itemcf_sim(df):
        user_item_time_dict = get_user_item_time(df)
        
        
        i2i_sim = {}
        item_cnt = defaultdict(int)
        for user, item_time_list in tqdm(user_item_time_dict.items()):
            # 在基于商品的协同过滤优化的时候可以考虑时间因素
            for i, i_click_time in item_time_list:
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for j, j_click_time in item_time_list:
                    if(i == j):
                        continue
                    i2i_sim[i].setdefault(j, 0)
                    
                    i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                    
        i2i_sim_ = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
        
        # 将得到的相似性矩阵保存到本地
        pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
        
        return i2i_sim_
    
    # 时间范围限定
    def _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_max=27, lag_hour_min=0):
    # 热度文章在用户最后一次点击时刻起，前3小时~27小时内的文章
        cur_time = time.time()
        lag_max = lag_hour_max * 60 * 60 * 1000
        lag_min = lag_hour_min * 60 * 60 * 1000
        # print(f'标点1时间是：{time.time()-cur_time}')
        # ts = articles_dic[articles_dic['article_id']==art_id]['created_at_ts'].values[0]
        ts = articles_dic.loc[art_id, 'created_at_ts']
        # print(f'文章时间：{ts}')
        # print(f'标点2时间是：{time.time()-cur_time}')
        if ts < (last_clicked_timestamp - lag_max):
            return False

        if ts > (last_clicked_timestamp - lag_min):
            return False
        # print(f'标点3时间是：{time.time()-cur_time}')

        
        return True

    # 物品协同召回
    def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,last_clicked_timestamp, articles_dic):
        # 获取用户历史交互的文章

        lag_hour_min = 0
        lag_hour_max = 27
        
        user_hist_items = user_item_time_dict[user_id]
        user_hist_items_ = {user_id for user_id, _ in user_hist_items}
        
        item_rank = {}
        # print('开始基于商品的召回i2i')
        for loc, (i, click_time) in enumerate(user_hist_items):
            for j, wij in i2i_sim[i][:sim_item_topk]:
                #j 是物品id
                if not _is_recall_target(last_clicked_timestamp, j, articles_dic, lag_hour_max=27, lag_hour_min=0):
                    continue
                if j in user_hist_items_:
                    continue

                item_rank.setdefault(j, 0)
                item_rank[j] +=  wij

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
            
        return item_rank

    ts = time.time()
    save_path='../results3/'

    # all_click_df, train_past_clicks, train_last_click, test_last_click = get_past_click()
    all_click_df=pd.read_csv(save_path+'all_click_df.csv')
    train_past_clicks = pd.read_csv(save_path+'train_past_clicks.csv')
    train_last_click = pd.read_csv(save_path+'train_last_click.csv')
    test_last_click = pd.read_csv(save_path+'test_last_click.csv')

    print('最后一次点击数据:')
    print(train_last_click[:5])
    #计算相似度
    i2i_sim = itemcf_sim(all_click_df)
    # 定义
    user_recall_items_dict = collections.defaultdict(dict)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(all_click_df)

    # 文章相似度字典
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

    #根据相似度从大到小排行
    for i in tqdm(i2i_sim.keys()):
        i2i_sim[i] = sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)
    
    # 相似文章的数量
    sim_item_topk = topk

    # 召回文章数量
    recall_item_num = topk
    articles_dic = articles.copy()
    articles_dic.set_index('article_id', inplace=True)
    train_last_click_time = train_last_click.set_index('user_id')['click_timestamp'].to_dict()
    
    print('文章召回开始')
    n1 = len(all_click_df['user_id'].unique())
    n2 = len(train_last_click['user_id'].unique())
    print(f'all_click_df:{n1}')
    print(f'last_click_df:{n2}')
    for user in tqdm(train_past_clicks['user_id'].unique()):
        
        click_time = train_last_click_time[user]
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, 
                                                            sim_item_topk, recall_item_num,click_time,articles_dic)
    # 将字典的形式转换成df
    user_item_score_list = []

    print('生成召回列表')
    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    recall_df.to_csv(save_path + 'recall_df.csv', index=False)

    # 从所有的召回数据中将测试集中的用户选出来
    tst_recall = recall_df[recall_df['user_id'].isin(test_last_click['user_id'].unique())]
    train_recall = recall_df[recall_df['user_id'].isin(train_last_click['user_id'].unique())]

    test_recall = tst_recall.copy()
    test_recall = test_recall.sort_values(by=['user_id', 'pred_score'])
    
    test_recall = test_recall.drop(columns=['pred_score'])

    test_recall.to_csv(save_path + 'itemcf_test_recall.csv', index=False)
    train_recall.to_csv(save_path + 'itemcf_train_recall.csv', index=False)

    print('Itemcf Recall Finished! Cost time: {}'.format(time.time() - ts))


    # return train_past_clicks, train_last_click, test_last_click