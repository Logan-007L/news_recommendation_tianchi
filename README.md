
# 解决方案
基于官方baseline物品协同过滤进行优化，得分: 0.1==>0.2==>0.23==>0.29

主要思路：

1. 先提高召回率，于是可以考虑到多路召回；

2. 用基础的lgbrank模型打印特征重要性可以知道新闻推荐对时效性较为敏感，官方也有给出一些统计信息的方法，于是可以针对时间特征对召回策略、特征工程进行优化

召回策略： 基于时效的物品协同过滤、热点文章召回和基于word2vec模型的相似文章召回。

特征工程：构造用户活跃度特征：基于点击次数和时间间隔，归一化计算活跃度得分，构造文章热度特征：通过时间窗口内点击次数和间隔归一化，反映文章热度。

排序：使用LGBMRanker对召回结果进行排序并取topk5输出

# 复现：
```
cd code
python main.py
```
