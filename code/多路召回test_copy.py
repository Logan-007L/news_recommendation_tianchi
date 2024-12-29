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
from data import get_past_click

warnings.filterwarnings('ignore')
ts = time.time()
data_path = '../data/' # 天池平台路径
save_path = '../results3/'  # 天池平台路径
# def get_past_click(train_click, test_click):
#     train = train_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
#     list1 = []
#     train_indexs = []

#     print('获取用户最后一次点击记录')
#     for user_id in tqdm(train['user_id'].unique()):
#         user = train[train['user_id'] == user_id]
#         #表示用户最后一次点击的记录
#         row = user.tail(1)
#         #最后点击的索引
#         train_indexs.append(row.index.values[0])
#         #testA中有一些只点了一次的用户要去掉
#         if len(user) >= 2:
#             list1.append(row.values.tolist()[0])
#     train_last_click = pd.DataFrame(list1, columns=['index', 'user_id', 'article_id', 'click_timestamp', 'click_environment',\
#                                     'click_deviceGroup', 'click_os', 'click_country', 'click_region',
#                                     'click_referrer_type'])
    
#     #最好一次点击的数据，做为验证集
#     train_last_click = train_last_click.drop(columns=['index'])

#     #除了最后一次点击的数据，做为训练集
#     train_past_clicks = train[~train.index.isin(train_indexs)]
#     train_past_clicks = train_past_clicks.drop(columns=['index'])
    
#     test = test_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
#     list2 = []
#     print('测试集获取用户最后一次点击记录')
#     for user_id in tqdm(test['user_id'].unique()):
#         user = test[test['user_id'] == user_id]
#         row = user.tail(1)
#         list2.append(row.values.tolist()[0])
#     test_last_click = pd.DataFrame(list2, columns=['index', 'user_id', 'article_id', 'click_timestamp', 'click_environment',\
#                                     'click_deviceGroup', 'click_os', 'click_country', 'click_region',
#                                     'click_referrer_type'])
#     test_last_click = test_last_click.drop(columns=['index'])
    
#     ###                    注释要去掉↓
#     all_click_df = pd.concat([train_past_clicks,test_click],ignore_index=True)
#     all_click_df = all_click_df.reset_index().drop(columns=['index'])

#     all_click_df = all_click_df.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

#     #保存结果
#     all_click_df.to_csv(save_path+'all_click_df.csv', index=False)
#     train_past_clicks.to_csv(save_path+'train_past_clicks.csv', index=False)
#     train_last_click.to_csv(save_path+'train_last_click.csv',index=False)
#     test_last_click.to_csv(save_path+'test_last_click.csv',index=False)
#     # return all_click_df, train_past_clicks, train_last_click, test_last_click

def submit_f(recall_df, topk=10, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())

    for i,_ in enumerate(tmp):
        if _ < topk:
            print(f'Warning: user {i} only has {_} articles, less than {topk}.')
    # assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]

    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d-%H-%M') + '.csv'
    submit.to_csv(save_name, index=False, header=True)


new_start=True
debug = True
articles = pd.read_csv(data_path + 'articles.csv')

#召回通道
itemcf = True
hot = True
word2vec = False
#召回数设置
itemcf_topk = 25
hot_topk = 10

#重新获取召回
if new_start:
    # # 全量训练集
    train_click = pd.read_csv(data_path + 'train_click_log.csv')
    testA_click = pd.read_csv(data_path + 'testA_click_log.csv')
    testB_click = pd.read_csv(data_path + 'testB_click_log.csv')

    if debug:
        train_click = pd.read_csv(data_path + 'train_click_log.csv').head(20000)
        testA_click = pd.read_csv(data_path + 'testA_click_log.csv').head(5000)
        testB_click = pd.read_csv(data_path + 'testB_click_log.csv').head(5000)
    
    train_click = pd.concat([train_click, testA_click,testB_click], ignore_index=True)
    test_click = pd.concat([testA_click, testB_click], ignore_index=True)
    
    #数据处理,获取past_click,last_click保存为csv文件
    get_past_click(train_click,test_click)

    all_click_df=pd.read_csv(save_path+'all_click_df.csv')
    train_past_clicks = pd.read_csv(save_path+'train_past_clicks.csv')
    train_last_click = pd.read_csv(save_path+'train_last_click.csv')
    test_last_click = pd.read_csv(save_path+'test_last_click.csv')

    #新闻相似度召回
    if itemcf:
        print('物品相似度召回')
        itemcf_recall(itemcf_topk,articles)
    #新闻热度召回
    if hot:
        print('热点新闻召回')
        hot_recall(hot_topk, train_click,test_click,train_past_clicks, test_last_click,articles)
    
    #word2vec召回
    if word2vec:
        print('word2vec召回')
        

    train = get_train_recall(itemcf, hot, train_last_click)
else:
    #直接读取
    train_past_clicks = pd.read_csv(save_path+'train_past_clicks.csv')
    train_last_click = pd.read_csv(save_path+'train_last_click.csv')
    test_last_click = pd.read_csv(save_path+'test_last_click.csv')
    #保存本地后直接读取
    train = pd.read_csv(save_path+'train_recall.csv')
    print('Train Total RECALL:{}%'.format((train['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))
    print('Train Total Recall Finished!')
    
train_past_clicks = train_past_clicks.groupby('user_id').agg({'click_timestamp': 'max'})

#第一次召回时执行下面的代码

print(f'训练集特征:{train.columns}')
#下采样
train = neg_sample(train)
print(train[:5])


#处理数据
train = train.sort_values('user_id').drop(columns=['click_timestamp']).reset_index(drop=True)
train = train.drop(columns=['click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']).merge(train_last_click.drop(columns=['article_id', 'click_timestamp']))
train = train.merge(articles, on='article_id', how='left')
train = train.merge(train_past_clicks, on='user_id', how='left')
train['delta_time'] = train['created_at_ts'] - train['click_timestamp']
# print(train.columns)


#数据集分割

X = train.copy()
y = train['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
X_eval, X_off, y_eval, y_off = train_test_split(X_test, y_test, test_size=0.5, random_state=66)

#g_train 存储的是每个用户在 X_train 数据集中对应标签（label）的数量。
g_train = X_train.groupby(['user_id'], as_index=False).count()['label'].values
g_eval = X_eval.groupby(['user_id'], as_index=False).count()['label'].values

from lightgbm import log_evaluation, early_stopping

lgb_cols = ['click_environment','click_deviceGroup', 'click_os', 'click_country', 
            'click_region','click_referrer_type', 'category_id', 'created_at_ts', 'words_count', 'click_timestamp', 'delta_time']

lgb_ranker = lgb.LGBMRanker(
    boosting_type='gbdt', 
    num_leaves=31, 
    reg_alpha=0.0, 
    reg_lambda=1,          
    max_depth=-1, 
    n_estimators=1000, 
    subsample=0.7, 
    colsample_bytree=0.7, 
    subsample_freq=1,   
    learning_rate=0.01, 
    min_child_weight=50, 
    random_state=66, 
    n_jobs=-1)


#定义早停止函数
callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

lgb_ranker.fit(
    X_train[lgb_cols], y_train, 
    group=g_train, 
    eval_set=[(X_eval[lgb_cols], y_eval)], 
    eval_group=[g_eval], 
    callbacks=callbacks,
)

#输出特征重要度
def print_feature_importance(columns, scores):
    print('--------------------------------')
    result = list(zip(columns, scores))
    result.sort(key=lambda v: v[1], reverse=True)
    for col, score in result:
        print('{}: {}'.format(col, score))
    print('--------------------------------')

print_feature_importance(lgb_cols, lgb_ranker.feature_importances_)

X_off['pred_score'] = lgb_ranker.predict(X_off[lgb_cols], num_iteration=lgb_ranker.best_iteration_)
X_off = X_off.drop(columns=['category_id', 'created_at_ts', 'words_count', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type', 'click_timestamp', 'delta_time'])
recall_df = X_off.copy()
recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

del recall_df['pred_score'], recall_df['label']
submit = recall_df[recall_df['rank'] <= 5].set_index(['user_id', 'rank']).unstack(-1).reset_index()
max_article = int(recall_df['rank'].value_counts().index.max())
submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]


#测试集结果生成
ts = time.time()
offline = False
if not offline:
    test_recall = get_test_recall(itemcf, hot)
    # test_recall = pd.read_csv(save_path+'test_recall.csv')
    test_recall = test_recall.merge(test_last_click.drop(columns=['article_id']))
    test_recall = test_recall.merge(articles, on='article_id', how='left')
    test_recall['delta_time'] = test_recall['created_at_ts'] - test_recall['click_timestamp']
    test_recall['pred_score'] = lgb_ranker.predict(test_recall[lgb_cols], num_iteration=lgb_ranker.best_iteration_)
    result = test_recall.sort_values(by=['user_id', 'pred_score'], ascending=(True, False))
    print(result.columns)
    result = result.drop(columns=['category_id', 'created_at_ts', 'words_count', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type', 'click_timestamp', 'delta_time'])
#         result.to_csv(save_path + 'test.csv', index=False)
    # 生成提交文件

    # 检测重复行
    duplicates = result[result.duplicated(keep=False)]  # keep=False 标记所有重复的行

    # 打印重复的行
    print("以下是重复的行：")
    print(duplicates)
    # 如果只需要统计重复行的数量：
    num_duplicates = duplicates.shape[0]
    print(f"重复行的数量为：{num_duplicates}")


    submit_f(result, topk=4, model_name='lgb_ranker')
    print('Submit Finished! Cost time: {}'.format(time.time() - ts))