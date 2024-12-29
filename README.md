# news_recommendation_tianchi
news_recommendation_tianchi

                \item \textbf{召回策略：} 设计基于时效的物品协同过滤、热点文章召回和基于word2vec模型的相似文章召回。
                \item \textbf{特征工程：}构造用户活跃度特征：基于点击次数和时间间隔，归一化计算活跃度得分，构造文章热度特征：通过时间窗口内点击次数和间隔归一化，反映文章热度。
                \item \textbf{排序：} 设计负采样策略，使用LgbRank对召回结果进行排序并取topk5输出
