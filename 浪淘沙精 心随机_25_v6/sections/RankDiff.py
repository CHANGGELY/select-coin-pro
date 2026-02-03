"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

# 第一步：计算每个时间点，每个币的"成交额排名"
def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['rank'] = df.groupby('candle_begin_time')[f'QuoteVolumeMean_{n}'].rank(ascending=True, method='min')
    df['rank_diff'] = df.groupby('symbol')['rank'].diff(n)

    df[factor_name] = df['rank_diff']#rank_diff 越小（越消极、负得越多）的币排在前面 # 意味着排名从 高(100) 掉到了 低(1)，代表成交额剧烈萎缩

    return df

# 第二步：计算排名的变化（当前排名 - n小时前的排名）
def get_factor_list(n):
    return [
        ('QuoteVolumeMean', n)
    ]
