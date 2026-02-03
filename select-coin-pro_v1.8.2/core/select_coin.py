"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import gc
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import job_num, factor_col_limit
from core.model.backtest_config import BacktestConfig, StrategyConfig
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import logger
from core.utils.path_kit import get_file_path

warnings.filterwarnings('ignore')
# pandas相关的显示设置，基础课程都有介绍
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)

# 计算完因子之后，保留的字段
KLINE_COLS = ['candle_begin_time', 'symbol', 'is_spot', 'close', 'next_close', 'symbol_spot', 'symbol_swap', '是否交易']
# 计算完选币之后，保留的字段
SELECT_RES_COLS = [*KLINE_COLS, 'strategy', 'cap_weight', '方向', 'offset', 'target_alloc_ratio', 'order_first']
# 完整kline数据保存的路径
ALL_KLINE_PATH_TUPLE = ('data', 'cache', 'all_factors_kline.pkl')
ALL_KLINE_FULL_PATH_TUPLE = ('data', 'cache', 'all_factors_kline_full.pkl')


# ======================================================================================
# 因子计算相关函数
# - calc_factors_by_symbol: 计算单个币种的因子池
# - calc_factors: 计算因子池
# ======================================================================================

def trans_period_for_day(df, date_col='candle_begin_time', factor_dict=None):
    """
    将数据周期转换为指定的1D周期
    :param df: 原始数据
    :param date_col: 日期列
    :param factor_dict: 转换规则
    :return:
    """
    df.set_index(date_col, inplace=True)
    # 必备字段
    agg_dict = {
        'symbol': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trade_num': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum',
        'is_spot': 'last',
        # 'has_swap': 'last',
        'symbol_swap': 'last',
        'symbol_spot': 'last',
        'funding_fee': 'sum',
        'next_avg_price': 'last',
        '是否交易': 'last',
    }

    if factor_dict:
        agg_dict = dict(agg_dict, **factor_dict)
    df = df.resample('1D').agg(agg_dict)
    df.reset_index(inplace=True)

    return df


# region 因子计算相关函数
def calc_factors_by_candle(candle_df, conf: BacktestConfig, factor_col_name_list) -> pd.DataFrame:
    """
    针对单一比对，计算所有因子的数值
    :param candle_df: 一个币种的k线数据 dataframe
    :param conf: 回测配置
    :param factor_col_name_list: 需要计算的因子列
    :return: 包含所有因子的 dataframe(目前是包含k线数据的）
    """
    # 遍历每个因子，计算每个因子的数据
    factor_series_dict = {}
    for factor_name, param_list in conf.factor_params_dict.items():
        factor = FactorHub.get_by_name(factor_name)  # 获取因子信息
        if factor.is_cross:
            continue

        # 筛选一下需要计算的因子
        factor_param_list = []
        for param in param_list:
            factor_col_name = f'{factor_name}_{param}'
            if factor_col_name in factor_col_name_list:
                factor_param_list.append(param)
        if len(factor_param_list) == 0:
            continue  # 当该因子不需要计算的时候直接返回

        # 如果存在外部数据，则使用 data_bridge 中的加载函数 load 数据
        if hasattr(factor, 'extra_data_dict') and factor.extra_data_dict:
            from core.utils.functions import merge_data
            for data_name in factor.extra_data_dict.keys():
                extra_data_dict = merge_data(candle_df, data_name, factor.extra_data_dict[data_name])
                for extra_data_name, extra_data_series in extra_data_dict.items():
                    candle_df[extra_data_name] = extra_data_series.values

        # 根据因子内部的函数，来判断是否进行加速操作
        if hasattr(factor, 'signal_multi_params'):  # 如果存在 signal_multi_params ，使用最新的因子加速写法
            result_dict = factor.signal_multi_params(candle_df, factor_param_list)
            for param, factor_series in result_dict.items():
                factor_series_dict[f'{factor_name}_{param}'] = factor_series.values

        else:  # 如果存在 signal，使用之前的老因子写法
            legacy_candle_df = candle_df.copy()  # 如果是老的因子计算逻辑，单独拿出来一份数据
            for param in factor_param_list:
                factor_col_name = f'{factor_name}_{param}'
                legacy_candle_df = factor.signal(legacy_candle_df, param, factor_col_name)
                factor_series_dict[factor_col_name] = legacy_candle_df[factor_col_name].values

    # 将结果 DataFrame 与原始 DataFrame 合并
    kline_with_factor_dict = {
        'candle_begin_time': candle_df['candle_begin_time'].values,
        'symbol': candle_df['symbol'].values,
        'is_spot': candle_df['is_spot'].values,
        'close': candle_df['close'].values,
        # 'has_swap': candle_df['has_swap'],
        # 'next_avg_price': candle_df['next_avg_price'].values,
        'next_close': candle_df['close'].shift(-1).values,  # 后面周期排除需要用
        # 'next_funding_fee': candle_df['funding_fee'].shift(-1).values,
        'symbol_spot': candle_df['symbol_spot'].astype(str).values,
        'symbol_swap': candle_df['symbol_swap'].astype(str).values,
        **factor_series_dict,
        '是否交易': candle_df['是否交易'].values,
    }

    kline_with_factor_df = pd.DataFrame(kline_with_factor_dict, copy=False)
    kline_with_factor_df.sort_values(by='candle_begin_time', inplace=True)

    # 抛弃一开始的一段k线，保留后面的数据
    first_candle_time = candle_df.iloc[0]['first_candle_time'] + pd.to_timedelta(f'{conf.min_kline_num}h')

    # 调整 symbol_spot 和 symbol_swap
    # for col in ['symbol_spot', 'symbol_swap']:
    #     symbol_start_time = candle_df[
    #         (candle_df[col] != '') & (candle_df[col].shift(1) == '') & (~candle_df[col].shift(1).isna())
    #         ]['candle_begin_time']
    #     if not symbol_start_time.empty:
    #         condition = pd.Series(False, index=kline_with_factor_df.index)
    #         for symbol_time in symbol_start_time:
    #             _cond1 = kline_with_factor_df['candle_begin_time'] > symbol_time
    #             _cond2 = kline_with_factor_df['candle_begin_time'] <= symbol_time + pd.to_timedelta(
    #                 f'{conf.min_kline_num}h')
    #             condition |= (_cond1 & _cond2)
    #         kline_with_factor_df.loc[condition, col] = ''
    #     kline_with_factor_df[col] = kline_with_factor_df[col].astype('category')

    # 需要对数据进行裁切
    kline_with_factor_df = kline_with_factor_df[kline_with_factor_df['candle_begin_time'] >= first_candle_time]

    # 下架币/拆分币，去掉最后一个周期不全的数据
    if kline_with_factor_df['candle_begin_time'].max() < pd.to_datetime(conf.end_date):
        _temp_time = kline_with_factor_df['candle_begin_time'] + pd.Timedelta(conf.max_hold_period)
        _del_time = kline_with_factor_df[kline_with_factor_df.loc[_temp_time.index, 'next_close'].isna()][
            'candle_begin_time']
        kline_with_factor_df = kline_with_factor_df[
            kline_with_factor_df['candle_begin_time'] <= _del_time.min() - pd.Timedelta(conf.max_hold_period)]

    # 只保留最近的数据
    if not conf.has_section_factor:
        kline_with_factor_df = kline_with_factor_df[
            (kline_with_factor_df['candle_begin_time'] >= pd.to_datetime(conf.start_date)) &
            (kline_with_factor_df['candle_begin_time'] < pd.to_datetime(conf.end_date))]

    # 只保留需要的字段
    return kline_with_factor_df


def process_candle_df(candle_df: pd.DataFrame, conf: BacktestConfig, factor_col_name_list: List[str], idx: int):
    """
    # 针对每一个币种的k线数据，按照策略循环计算因子信息
    :param candle_df: 单个币种的数据
    :param conf: backtest config
    :param factor_col_name_list:    因子列表，可以用于动态判断当前需要计算的因子列。
                                    当 factor_col_name_list ≠ conf.factor_col_name_list 时，说明需要节省一点内存
    :param idx: 索引
    :return: 带有因子数值的数据
    """
    # ==== 数据预处理 ====
    factor_dict = {'first_candle_time': 'first', 'last_candle_time': 'last'}
    for strategy in conf.strategy_list:
        symbol = candle_df['symbol'].iloc[-1]
        candle_df, _factor_dict, _ = strategy.after_merge_index(candle_df, symbol, factor_dict, {})
        factor_dict.update(_factor_dict)

    # 计算平均开盘价格
    candle_df['next_avg_price'] = candle_df[conf.avg_price_col].shift(-1)  # 用于后面计算当周期涨跌幅

    # 转换成日线数据  跟回测保持一致
    if conf.is_day_period:
        candle_df = trans_period_for_day(candle_df, factor_dict=factor_dict)

    # ==== 计算因子 ====
    # 清理掉头部参与日线转换的填充数据
    candle_df.dropna(subset=['symbol'], inplace=True)
    candle_df.reset_index(drop=True, inplace=True)
    # 针对单个币种的K线数据计算
    # 返回带有因子数值的K线数据
    factor_df = calc_factors_by_candle(candle_df, conf, factor_col_name_list)

    return idx, factor_df


def calc_factors(conf: BacktestConfig):
    """
    选币因子计算，考虑到大因子回测的场景，我们引入chunk的概念，会把所有factor切成多分，然后分别计算
    :param conf:       账户信息
    :return:
    """
    # ====================================================================================================
    # 1. ** k线数据整理及参数准备 **
    # - is_use_spot: True的时候，使用现货数据和合约数据;
    # - False的时候，只使用合约数据。所以这个情况更简单
    # ====================================================================================================
    # hold_period的作用是计算完因子之后，
    # 获取最近 hold_period 个小时内的数据信息，
    # 同时用于offset字段计算使用
    # ====================================================================================================
    # 2. ** 因子计算 **
    # 遍历每个币种，计算相关因子数据
    # ====================================================================================================
    candle_df_list = pd.read_pickle(get_file_path('data', 'cache', 'all_candle_df_list.pkl'))
    num_candles = len(candle_df_list)  # 内存优化: 缓存长度
    factor_col_count = len(conf.factor_col_name_list)
    shards = range(0, factor_col_count, factor_col_limit)

    logger.debug(f'''* 总共计算因子个数：{factor_col_count} 个
* 单次计算因子个数：{factor_col_limit} 个，(需分成{len(shards)}组计算)
* 需要计算币种数量：{num_candles} 个''')

    # ====================================================================================================
    # 智能因子缓存检测
    # 检查所有需要的因子 pkl 是否已存在，存在则跳过计算
    # ====================================================================================================
    from config import use_cache
    all_kline_pkl = get_file_path(*ALL_KLINE_PATH_TUPLE, as_path_type=True)
    all_kline_full_pkl = get_file_path(*ALL_KLINE_FULL_PATH_TUPLE, as_path_type=True)

    if use_cache:
        # 检查 K 线缓存
        kline_cache_valid = all_kline_pkl.exists()
        if conf.has_section_factor:
            kline_cache_valid = kline_cache_valid and all_kline_full_pkl.exists()

        # 检查因子缓存
        missing_factors = []
        for factor_col_name in conf.factor_col_name_list:
            factor_pkl = get_file_path('data', 'cache', f'factor_{factor_col_name}.pkl', as_path_type=True)
            if not factor_pkl.exists():
                missing_factors.append(factor_col_name)

        if kline_cache_valid and len(missing_factors) == 0:
            logger.debug('✅ 因子缓存有效，跳过因子计算阶段')
            # 释放内存后直接返回（跳过后续的因子计算循环）
            del candle_df_list
            gc.collect()
            return  # 直接结束函数，后续选币阶段会从缓存读取

        if len(missing_factors) > 0:
            logger.debug(f'📋 需要计算 {len(missing_factors)} 个缺失因子: {missing_factors[:5]}...')
    else:
        # 禁用缓存时，清理旧文件
        all_kline_pkl.unlink(missing_ok=True)
        all_kline_full_pkl.unlink(missing_ok=True)

    for shard_index in shards:
        logger.debug(f'🚀 因子分片计算中，进度：{int(shard_index / factor_col_limit) + 1}/{len(shards)}')
        factor_col_name_list = conf.factor_col_name_list[shard_index:shard_index + factor_col_limit]

        all_factor_df_list = [pd.DataFrame()] * num_candles
        with ProcessPoolExecutor(max_workers=job_num) as executor:
            futures = [executor.submit(
                process_candle_df, candle_df.copy(), conf, factor_col_name_list, candle_idx
            ) for candle_idx, candle_df in enumerate(candle_df_list)]

            for future in tqdm(as_completed(futures), total=num_candles, desc='🧮 时序因子计算'):
                idx, factor_df = future.result()
                all_factor_df_list[idx] = factor_df

        # ====================================================================================================
        # 3. ** 合并因子结果 **
        # 合并并整理所有K线，到这里因子计算完成
        # ====================================================================================================
        # 内存优化: 分批合并，避免一次性分配大内存
        logger.debug('🔗 合并因子结果...')
        all_factors_df = pd.concat(all_factor_df_list, ignore_index=True, copy=False)
        all_factors_df['symbol'] = pd.Categorical(all_factors_df['symbol'])

        del all_factor_df_list
        gc.collect()  # 内存优化: 及时回收

        # ====================================================================================================
        # 4. ** 因子结果分片存储 **
        # 分片存储计算结果，节省内存占用，提高选币效率
        # - 将合并好的df，分成2个部分：k线和因子列
        # - k线数据存储为一个pkl，每一列因子存储为一个pkl，在选币时候按需读入合并成df
        # ====================================================================================================
        logger.debug('💾 分片存储因子结果...')

        # 选币需要的k线
        if not all_kline_pkl.exists():
            # 存储裁切时间的数据
            all_kline_df = all_factors_df[KLINE_COLS].sort_values(by=['candle_begin_time', 'symbol', 'is_spot'])
            all_kline_df = all_kline_df[
                (all_kline_df['candle_begin_time'] >= pd.to_datetime(conf.start_date)) &
                (all_kline_df['candle_begin_time'] < pd.to_datetime(conf.end_date))]
            all_kline_df.to_pickle(all_kline_pkl)
            del all_kline_df  # 内存优化
            gc.collect()

        if not all_kline_full_pkl.exists() and conf.has_section_factor:
            # 存储不裁切的全量数据
            all_kline_full_df = all_factors_df[KLINE_COLS].sort_values(by=['candle_begin_time', 'symbol', 'is_spot'])
            all_kline_full_df.to_pickle(all_kline_full_pkl)
            del all_kline_full_df  # 内存优化
            gc.collect()

        # 针对每一个因子进行存储
        cut_factors_df = all_factors_df[
                (all_factors_df['candle_begin_time'] >= pd.to_datetime(conf.start_date)) &
                (all_factors_df['candle_begin_time'] < pd.to_datetime(conf.end_date))]
        for factor_col_name in factor_col_name_list:
            factor_pkl = get_file_path('data', 'cache', f'factor_{factor_col_name}.pkl', as_path_type=True)
            factor_pkl.unlink(missing_ok=True)  # 动态清理掉cache的缓存
            # 截面因子数据不在这里计算，不存在这个列名
            if factor_col_name not in all_factors_df.columns:
                continue

            if conf.has_section_factor:
                factor_full_pkl = get_file_path('data', 'cache', f'factor_full_{factor_col_name}.pkl', as_path_type=True)
                factor_full_pkl.unlink(missing_ok=True)  # 动态清理掉cache的缓存

                # 存储不裁切的全量数据
                all_factors_df[factor_col_name].to_pickle(factor_full_pkl)

            cut_factors_df[factor_col_name].to_pickle(factor_pkl)

        del all_factors_df, cut_factors_df
        gc.collect()  # 内存优化: 每个分片结束后清理

    # 内存优化: 循环结束后再删除大数据
    del candle_df_list
    gc.collect()


def process_factor_df(factor_col_name):
    # 准备所有时序因子数据
    factor_path = get_file_path('data', 'cache', f'factor_full_{factor_col_name}.pkl', as_path_type=True)
    if not factor_path.exists():
        return factor_col_name, pd.DataFrame()

    return factor_col_name, pd.read_pickle(factor_path)


def load_all_factors(conf: BacktestConfig):
    all_kline_full_pkl = get_file_path(*ALL_KLINE_FULL_PATH_TUPLE, as_path_type=True)
    factor_df = pd.read_pickle(all_kline_full_pkl)

    # 准备所有时序因子数据
    with ProcessPoolExecutor(max_workers=job_num) as executor:
        futures = [executor.submit(
            process_factor_df, factor_col_name
        ) for factor_col_name in conf.section_depend_factor_col_name_list]

        for future in tqdm(as_completed(futures), total=len(conf.section_depend_factor_col_name_list), desc='✂️ 裁切时序因子数据'):
            factor_col_name, kline_with_factor_df = future.result()
            if not kline_with_factor_df.empty:
                factor_df[factor_col_name] = kline_with_factor_df

    return factor_df


def calc_cross_sections(conf: BacktestConfig):
    """
    截面因子计算，
    :param conf:       账户信息
    :return:
    """
    section_params_dict = conf.section_params_dict
    # 如果没有配置截面因子，那么直接跳过后续
    if not section_params_dict:
        logger.info(f'未检查到截面因子配置，跳过计算截面因子步骤。')
        return

    # ====================================================================================================
    # 智能截面因子缓存检测
    # 检查所有截面因子的 pkl 是否已存在，存在则跳过计算（避免加载庞大的面板数据）
    # ====================================================================================================
    from config import use_cache
    if use_cache:
        missing_cross_factors = []
        for factor_name, param_list in section_params_dict.items():
            for param in param_list:
                factor_col_name = f'{factor_name}_{param}'
                if factor_col_name in conf.factor_col_name_list:
                    factor_pkl = get_file_path('data', 'cache', f'factor_{factor_col_name}.pkl', as_path_type=True)
                    if not factor_pkl.exists():
                        missing_cross_factors.append(factor_col_name)

        if len(missing_cross_factors) == 0:
            logger.debug('✅ 截面因子缓存有效，跳过计算')
            return

        logger.debug(f'📋 需要计算 {len(missing_cross_factors)} 个缺失截面因子')

    # 加载面板数据（仅在需要计算时才加载）
    factor_df = load_all_factors(conf)

    # condition = factor_df['is_spot'] == (1 if conf.is_use_spot else 0)
    # factor_df = factor_df.loc[condition, :]
    # factor_spot_df = factor_df.loc[factor_df['is_spot'] == 1, :].copy()
    # factor_swap_df = factor_df.loc[factor_df['is_spot'] == 0, :].copy()

    # 遍历截面因子，调用截面因子计算方法
    for factor_name, param_list in section_params_dict.items():
        factor = FactorHub.get_by_name(factor_name)  # 获取因子信息

        # 筛选一下需要计算的因子
        factor_param_list = []
        section_param_list = []
        for param in param_list:
            factor_col_name = f'{factor_name}_{param}'
            if factor_col_name in conf.factor_col_name_list:
                factor_param_list.append(param)
                section_param_list.extend(factor.get_factor_list(param))
        if len(factor_param_list) == 0:
            continue  # 当该因子不需要计算的时候直接返回

        # 截面因子依赖的时序因子列
        section_col_name_list = list(set(f'{f}_{n}' for f, n in set(section_param_list)))

        # 对截面因子按照时间进行分段计算
        legacy_candle_df = factor_df[KLINE_COLS + section_col_name_list].copy()  # 如果是老的因子计算逻辑，单独拿出来一份数据
        for param in tqdm(factor_param_list, total=len(factor_param_list), desc=f'🧮 截面因子计算'):
            factor_col_name = f'{factor_name}_{param}'
            legacy_candle_df = factor.signal(legacy_candle_df, param, factor_col_name)

            # 对数据进行裁切并保存
            cross_factor_df = legacy_candle_df[['candle_begin_time', factor_col_name]]
            cross_factor_df = cross_factor_df[
                (cross_factor_df['candle_begin_time'] >= pd.to_datetime(conf.start_date)) &
                (cross_factor_df['candle_begin_time'] < pd.to_datetime(conf.end_date))]
            factor_pkl = get_file_path('data', 'cache', f'factor_{factor_col_name}.pkl', as_path_type=True)
            factor_pkl.unlink(missing_ok=True)  # 动态清理掉cache的缓存
            cross_factor_df[factor_col_name].to_pickle(factor_pkl)
            del cross_factor_df
        del legacy_candle_df

    del factor_df
    gc.collect()


# endregion


# ======================================================================================
# 选币相关函数
# - calc_select_factor_rank: 计算因子排序
# - select_long_and_short_coin: 选做多和做空的币种
# - select_coins_by_strategy: 根据策略选币
# - select_coins: 选币，循环策略调用 `select_coins_by_strategy`
# ======================================================================================
# region 选币相关函数
def calc_select_factor_rank(df, factor_column='因子', ascending=True):
    """
    计算因子排名
    :param df:              原数据
    :param factor_column:   需要计算排名的因子名称
    :param ascending:       计算排名顺序，True：从小到大排序；False：从大到小排序
    :return:                计算排名后的数据框
    """
    # 计算因子的分组排名
    df['rank'] = df.groupby('candle_begin_time')[factor_column].rank(method='min', ascending=ascending)
    df['rank_max'] = df.groupby('candle_begin_time')['rank'].transform('max')
    # 根据时间和因子排名排序
    df.sort_values(by=['candle_begin_time', 'rank'], inplace=True)
    # 重新计算一下总币数
    df['总币数'] = df.groupby('candle_begin_time')['symbol'].transform('size')
    return df


def select_long_and_short_coin(strategy: StrategyConfig, long_df: pd.DataFrame, short_df: pd.DataFrame):
    """
    选币，添加多空资金权重后，对于无权重的情况，减少选币次数

    :param strategy:                策略，包含：多头选币数量，空头选币数量，做多因子名称，做空因子名称，多头资金权重，空头资金权重
    :param long_df:                 多头选币的df
    :param short_df:                空头选币的df
    :return:
    """
    """
    # 做多选币
    """
    if strategy.long_cap_weight > 0:
        long_df = calc_select_factor_rank(long_df, factor_column=strategy.long_factor, ascending=True)

        long_df = strategy.select_by_coin_num(long_df, strategy.long_select_coin_num)

        long_df['方向'] = 1
        long_df['target_alloc_ratio'] = 1 / long_df.groupby('candle_begin_time')['symbol'].transform('size')
    else:
        long_df = pd.DataFrame()

    """
    # 做空选币
    """
    if strategy.short_cap_weight > 0:
        short_df = calc_select_factor_rank(short_df, factor_column=strategy.short_factor, ascending=False)

        if strategy.short_select_coin_num == 'long_nums':  # 如果参数是long_nums，则空头与多头的选币数量保持一致
            # 获取到多头的选币数量并整理数据
            long_select_num = long_df.groupby('candle_begin_time')['symbol'].size().to_frame()
            long_select_num = long_select_num.rename(columns={'symbol': '多头数量'}).reset_index()
            # 将多头选币数量整理到short_df
            short_df = short_df.merge(long_select_num, on='candle_begin_time', how='left')
            # 使用多头数量对空头数据进行选币
            short_df = short_df[short_df['rank'] <= short_df['多头数量']]
            del short_df['多头数量']
        else:
            short_df = strategy.select_by_coin_num(short_df, strategy.short_select_coin_num)

        short_df['方向'] = -1
        short_df['target_alloc_ratio'] = 1 / short_df.groupby('candle_begin_time')['symbol'].transform('size')
    else:
        short_df = pd.DataFrame()

    # ===整理数据
    df = pd.concat([long_df, short_df], ignore_index=True)  # 将做多和做空的币种数据合并
    df.sort_values(by=['candle_begin_time', '方向'], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    del df['总币数'], df['rank_max']

    return df


def select_coins_by_strategy(factor_df, stg_conf: StrategyConfig):
    """
    针对每一个策略，进行选币，具体分为以下4步：
    - 4.1 数据清洗
    - 4.2 计算目标选币因子
    - 4.3 前置过滤筛选
    - 4.4 根据选币因子进行选币
    :param stg_conf: 策略配置
    :param factor_df: 所有币种K线数据，仅包含部分行情数据和选币需要的因子列
    :return: 选币数据
    """

    """
    4.1 数据预处理
    可以预留一些空间给数据整理，比如缺失数据的处理
    """
    pass

    """
    4.2 计算目标选币因子
    """
    s = time.time()
    # 缓存计算前的列名
    prev_cols = factor_df.columns
    # 计算因子
    result_df = stg_conf.calc_select_factor(factor_df)
    # 合并新的因子
    factor_df = factor_df[prev_cols].join(result_df[list(set(result_df.columns) - set(prev_cols))])
    logger.debug(f'[{stg_conf.name}] 选币因子计算耗时：{time.time() - s:.2f}s')

    """
    4.3 前置过滤筛选
    """
    s = time.time()
    long_df, short_df = stg_conf.filter_before_select(factor_df)
    short_df = short_df[short_df['symbol_swap'] != '']  # 保留有合约的现货
    logger.debug(f'[{stg_conf.name}] 前置过滤耗时：{time.time() - s:.2f}s')

    """
    4.4 根据选币因子进行选币
    """
    s = time.time()
    # 多头选币数据、空头选币数据、策略配置
    factor_df = select_long_and_short_coin(stg_conf, long_df, short_df)
    logger.debug(f'[{stg_conf.name}] 多空选币耗时：{time.time() - s:.2f}s')

    """
    4.5 后置过滤筛选
    """
    factor_df = stg_conf.filter_after_select(factor_df)
    logger.debug(f'[{stg_conf.name}] 后置过滤耗时：{time.time() - s:.2f}s')

    """
    4.6 根据多空比调整币种的权重
    """
    long_ratio = stg_conf.long_cap_weight / (stg_conf.long_cap_weight + stg_conf.short_cap_weight)
    factor_df.loc[factor_df['方向'] == 1, 'target_alloc_ratio'] = factor_df['target_alloc_ratio'] * long_ratio
    factor_df.loc[factor_df['方向'] == -1, 'target_alloc_ratio'] = factor_df['target_alloc_ratio'] * (1 - long_ratio)
    factor_df = factor_df[factor_df['target_alloc_ratio'].abs() > 1e-9]  # 去除权重为0的数据

    return factor_df[[*KLINE_COLS, '方向', 'target_alloc_ratio']]


def process_strategy(stg_conf: StrategyConfig, result_folder: Path):
    s = time.time()
    strategy_name = stg_conf.name
    logger.debug(f'[{stg_conf.name}] 开始选币...')

    # 准备选币用数据
    factor_df = pd.read_pickle(get_file_path(*ALL_KLINE_PATH_TUPLE))
    # 时序因子
    for factor_col_name in stg_conf.factor_columns:
        factor_df[factor_col_name] = pd.read_pickle(
            get_file_path('data', 'cache', f'factor_{factor_col_name}.pkl'))
    factor_df = factor_df[factor_df['是否交易'] == 1]

    select_scope = stg_conf.select_scope
    order_first = stg_conf.order_first
    if select_scope == 'spot':
        condition = (factor_df['is_spot'] == 1)
    elif select_scope == 'swap':
        condition = (factor_df['is_spot'] == 0)
    else:  # mix 混合
        both_not_null = (factor_df['symbol_spot'] != '') & (factor_df['symbol_swap'] != '')
        # 根据优先下单，处理选币的币种
        order_first_symbol = (factor_df['is_spot'] == (1 if order_first == 'spot' else 0))
        condition = (~both_not_null | order_first_symbol)
    factor_df = factor_df.loc[condition, :].copy()

    factor_df.dropna(subset=stg_conf.factor_columns, inplace=True)
    factor_df.dropna(subset=['symbol'], how='any', inplace=True)

    factor_df.sort_values(by=['candle_begin_time', 'symbol'], inplace=True)
    factor_df.reset_index(drop=True, inplace=True)

    logger.debug(f'[{stg_conf.name}] 选币数据准备完成，消耗时间：{time.time() - s:.2f}s')

    result_df = select_coins_by_strategy(factor_df, stg_conf)
    # 用于缓存选币结果，如果结果为空，也会生成对应的，空的pkl文件
    stg_select_result = result_folder / f'{stg_conf.get_fullname(as_folder_name=True)}.pkl'

    if result_df.empty:
        pd.DataFrame(columns=SELECT_RES_COLS).to_pickle(stg_select_result)
        return

    del factor_df

    # 筛选合适的offset
    cal_offset_base_seconds = 3600 * 24 if stg_conf.is_day_period else 3600
    reference_date = pd.to_datetime('2017-01-01')
    time_diff_seconds = (result_df['candle_begin_time'] - reference_date).dt.total_seconds()
    offset = (time_diff_seconds / cal_offset_base_seconds).mod(stg_conf.period_num).astype('int8')
    result_df['offset'] = ((offset + 1 + stg_conf.period_num) % stg_conf.period_num).astype('int8')
    result_df = result_df[result_df['offset'].isin(stg_conf.offset_list)]

    if result_df.empty:
        pd.DataFrame(columns=SELECT_RES_COLS).to_pickle(stg_select_result)
        return

    # 添加其他的相关选币信息
    select_result_dict = dict()
    for kline_col in KLINE_COLS:
        select_result_dict[kline_col] = result_df[kline_col].values

    select_result_dict['方向'] = result_df['方向'].astype('int8').values
    select_result_dict['offset'] = result_df['offset'].astype('int8').values
    select_result_dict['target_alloc_ratio'] = result_df['target_alloc_ratio'].values
    select_result_df = pd.DataFrame(select_result_dict, copy=False)
    del result_df

    select_result_df['strategy'] = strategy_name
    select_result_df['strategy'] = pd.Categorical(select_result_df['strategy'])

    # 根据策略资金权重，调整目标分配比例
    select_result_df['cap_weight'] = np.float64(stg_conf.cap_weight)
    select_result_df['target_alloc_ratio'] = np.float64(
        select_result_df['target_alloc_ratio']
        * select_result_df['cap_weight']
        / len(stg_conf.offset_list)
        * select_result_df['方向']
    )
    select_result_df['order_first'] = order_first

    # 缓存到本地文件
    select_result_df[SELECT_RES_COLS].to_pickle(stg_select_result)

    logger.debug(f'[{strategy_name}] 耗时: {(time.time() - s):.2f}s')

    gc.collect()


# 选币数据整理 & 选币
def select_coin_with_conf(conf: BacktestConfig, multi_process=True, silent=False):
    """
    ** 策略选币 **
    - is_use_spot: True的时候，使用现货数据和合约数据;
    - False的时候，只使用合约数据。所以这个情况更简单

    :param conf: 回测配置
    :param multi_process: 是否启用多进程
    :param silent: 是否静默
    :return:
    """
    if silent:
        import logging
        logger.setLevel(logging.WARNING)  # 可以减少中间输出的log
    # ====================================================================================================
    # 2.1 初始化
    # ====================================================================================================
    result_folder = conf.get_result_folder()  # 选币结果文件夹

    if not multi_process:
        for strategy in conf.strategy_list:
            process_strategy(strategy, result_folder)
        return

    # 多进程模式
    with ProcessPoolExecutor(max_workers=job_num) as executor:
        futures = [executor.submit(process_strategy, stg, result_folder) for stg in conf.strategy_list]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.exception(e)
                exit(1)


def select_coins(confs: BacktestConfig | List[BacktestConfig], multi_process=True):
    if isinstance(confs, BacktestConfig):
        # 如果是单例，就直接返回原来的结果
        return select_coin_with_conf(confs, multi_process=multi_process)

    # 否则就直接并行回测
    is_multi = False  # 怕资源溢出，强制串行
    is_silent = True  # 减少输出
    with ProcessPoolExecutor(max_workers=job_num) as executor:
        futures = [executor.submit(select_coin_with_conf, conf, is_multi, is_silent) for conf in confs]
        for future in tqdm(as_completed(futures), total=len(confs), desc='选币'):
            try:
                future.result()
            except Exception as e:
                logger.exception(e)
                exit(1)


# endregion

# ======================================================================================
# 选币结果聚合
# ======================================================================================
# region 选币结果聚合
def transfer_swap(select_coin, df_swap):
    """
    将现货中的数据替换成合约数据，主要替换：close
    :param select_coin:     选币数据
    :param df_swap:         合约数据
    :return:
    """
    trading_cols = ['symbol', 'is_spot', 'close', 'next_close']

    # 找到我们选币结果中，找到有对应合约的现货选币
    spot_line_index = select_coin[(select_coin['symbol_swap'] != '') & (select_coin['is_spot'] == 1)].index
    spot_select_coin = select_coin.loc[spot_line_index].copy()

    # 其他的选币，也就是要么已经是合约，要么是现货但是找不到合约
    swap_select_coin = select_coin.loc[select_coin.index.difference(spot_line_index)].copy()

    # 合并合约数据，找到对应的合约（原始数据不动，新增_2）
    # ['candle_begin_time', 'symbol_swap', 'strategy', 'cap_weight', '方向', 'offset', 'target_alloc_ratio']
    spot_select_coin = pd.merge(
        spot_select_coin, df_swap[['candle_begin_time', *trading_cols]],
        left_on=['candle_begin_time', 'symbol_swap'], right_on=['candle_begin_time', 'symbol'],
        how='left', suffixes=('', '_2'))

    # merge完成之后，可能因为有些合约数据上线不超过指定的时间（min_kline_num）,造成合并异常，需要按照原现货逻辑执行
    failed_merge_select_coin = spot_select_coin[spot_select_coin['close_2'].isna()][select_coin.columns].copy()

    spot_select_coin = spot_select_coin.dropna(subset=['close_2'], how='any')
    spot_select_coin['is_spot_2'] = spot_select_coin['is_spot_2'].astype(np.int8)

    spot_select_coin.drop(columns=trading_cols, inplace=True)
    rename_dict = {f'{trading_col}_2': trading_col for trading_col in trading_cols}
    spot_select_coin.rename(columns=rename_dict, inplace=True)

    # 将拆分的选币数据，合并回去
    # 1. 纯合约部分，或者没有合约的现货 2. 不能转换的现货 3. 现货被替换为合约的部分
    select_coin = pd.concat([swap_select_coin, failed_merge_select_coin, spot_select_coin], axis=0)
    select_coin.sort_values(['candle_begin_time', '方向'], inplace=True)

    return select_coin


def transfer_spot(select_coin, df_spot):
    """
    将现货中的数据替换成合约数据，主要替换：close
    :param select_coin:     选币数据
    :param df_spot:         合约数据
    :return:
    """
    trading_cols = ['symbol', 'is_spot', 'close', 'next_close']

    # 找到我们选币结果中，找到有对应合约的现货选币
    spot_line_index = select_coin[(select_coin['symbol_spot'] != '') & (select_coin['is_spot'] == 0)].index
    spot_select_coin = select_coin.loc[spot_line_index].copy()

    # 其他的选币，也就是要么已经是合约，要么是现货但是找不到合约
    swap_select_coin = select_coin.loc[select_coin.index.difference(spot_line_index)].copy()

    # 合并合约数据，找到对应的合约（原始数据不动，新增_2）
    # ['candle_begin_time', 'symbol_swap', 'strategy', 'cap_weight', '方向', 'offset', 'target_alloc_ratio']
    spot_select_coin = pd.merge(
        spot_select_coin, df_spot[['candle_begin_time', *trading_cols]],
        left_on=['candle_begin_time', 'symbol_spot'], right_on=['candle_begin_time', 'symbol'],
        how='left', suffixes=('', '_2'))

    # merge完成之后，可能因为有些合约数据上线不超过指定的时间（min_kline_num）,造成合并异常，需要按照原现货逻辑执行
    failed_merge_select_coin = spot_select_coin[spot_select_coin['close_2'].isna()][select_coin.columns].copy()

    spot_select_coin = spot_select_coin.dropna(subset=['close_2'], how='any')
    spot_select_coin['is_spot_2'] = spot_select_coin['is_spot_2'].astype(np.int8)

    spot_select_coin.drop(columns=trading_cols, inplace=True)
    rename_dict = {f'{trading_col}_2': trading_col for trading_col in trading_cols}
    spot_select_coin.rename(columns=rename_dict, inplace=True)

    # 将拆分的选币数据，合并回去
    # 1. 纯合约部分，或者没有合约的现货 2. 不能转换的现货 3. 现货被替换为合约的部分
    select_coin = pd.concat([swap_select_coin, failed_merge_select_coin, spot_select_coin], axis=0)
    select_coin.sort_values(['candle_begin_time', '方向'], inplace=True)

    return select_coin


def concat_select_results(conf: BacktestConfig) -> None:
    """
    聚合策略选币结果，形成综合选币结果
    :param conf:
    :return:
    """
    # 如果是纯多头现货模式，那么就不转换合约数据，只下现货单
    all_select_result_df_list = []  # 存储每一个策略的选币结果
    result_folder = conf.get_result_folder()
    select_result_path = result_folder / '选币结果.pkl'

    for strategy in conf.strategy_list:
        stg_select_result = result_folder / f'{strategy.get_fullname(as_folder_name=True)}.pkl'

        # 如果文件不存在，就跳过
        if not os.path.exists(stg_select_result):
            continue

        all_select_result_df_list.append(pd.read_pickle(stg_select_result))

    # 如果没有任何策略的选币结果，就直接返回
    if not all_select_result_df_list:
        pd.DataFrame(columns=SELECT_RES_COLS).to_pickle(select_result_path)
        return

    # 聚合选币结果
    all_select_result_df = pd.concat(all_select_result_df_list, ignore_index=True)
    del all_select_result_df_list
    gc.collect()
    all_select_result_df.to_pickle(select_result_path)


def process_select_results(conf: BacktestConfig) -> pd.DataFrame:
    select_result_path = conf.get_result_folder() / '选币结果.pkl'
    if not select_result_path.exists():
        logger.warning('没有生成选币文件，直接返回')
        return pd.DataFrame(columns=SELECT_RES_COLS)
    all_select_result_df = pd.read_pickle(select_result_path)

    # 筛选一下选币结果，判断其中的 优先下单标记是什么
    cond1 = all_select_result_df['order_first'] == 'swap'  # 优先下单合约
    cond2 = all_select_result_df['is_spot'] == 1  # 当前币种是现货
    if not all_select_result_df[cond1 & cond2].empty:
        # 如果现货部分有对应的合约，我们会把现货比对替换为对应的合约，来节省手续费（合约交易手续费比现货要低）
        all_kline_df = pd.read_pickle(get_file_path(*ALL_KLINE_PATH_TUPLE))
        # 将含有现货的币种，替换掉其中close价格
        df_swap = all_kline_df[(all_kline_df['is_spot'] == 0) & (all_kline_df['symbol_spot'] != '')]
        no_transfer_df = all_select_result_df[~(cond1 & cond2)]
        all_select_result_df = transfer_swap(all_select_result_df[cond1 & cond2], df_swap)
        all_select_result_df = pd.concat([no_transfer_df, all_select_result_df], ignore_index=True)

    cond1 = all_select_result_df['order_first'] == 'spot'  # 优先下单合约
    cond2 = all_select_result_df['is_spot'] == 0  # 当前币种是现货
    if not all_select_result_df[cond1 & cond2].empty:
        all_kline_df = pd.read_pickle(get_file_path(*ALL_KLINE_PATH_TUPLE))
        df_spot = all_kline_df[(all_kline_df['is_spot'] == 1) & (all_kline_df['symbol_swap'] != '')]
        no_transfer_df = all_select_result_df[~(cond1 & cond2)]
        all_select_result_df = transfer_spot(all_select_result_df[cond1 & cond2], df_spot)
        all_select_result_df = pd.concat([no_transfer_df, all_select_result_df], ignore_index=True)

    return all_select_result_df


def to_ratio_pivot(df_select: pd.DataFrame, candle_begin_times, columns) -> pd.DataFrame:
    # 转换为仓位比例，index 为时间，columns 为币种，values 为比例的求和
    df_ratio = df_select.pivot_table(
        index='candle_begin_time', columns=columns, values='target_alloc_ratio',
        fill_value=0, aggfunc='sum', observed=True
    )

    # 重新填充为完整的小时级别数据
    df_ratio = df_ratio.reindex(candle_begin_times, fill_value=0)
    return df_ratio


def trim_ratio_delists(df_ratio: pd.DataFrame, end_time: pd.Timestamp, market_dict: dict, trade_type: str):
    """
    ** 删除要下架的币 **
    当币种即将下架的时候，把后续的持仓调整为 0
    :param df_ratio: 仓位比例
    :param end_time: 回测结束时间
    :param market_dict: 所有币种的K线数据
    :param trade_type: spot or swap
    :return: 仓位调整后的比例
    """
    for symbol in df_ratio.columns:
        df_market = market_dict[symbol]
        if len(df_market) < 2:
            continue

        # 没有下架
        last_end_time = df_market['candle_begin_time'].iloc[-1]
        if last_end_time >= end_time:
            continue

        second_last_end_time = df_market['candle_begin_time'].iloc[-2]
        if (df_ratio.loc[second_last_end_time:, symbol].abs() > 1e-8).any():
            logger.warning(f'{trade_type} {symbol} 下架选币权重不为 0，清除 {second_last_end_time} 之后的权重')
            df_ratio.loc[second_last_end_time:, symbol] = 0

    return df_ratio


def agg_offset_by_strategy(df_select: pd.DataFrame, stg_conf: StrategyConfig):
    # 如果没有现货选币结果，就返回空
    if df_select.empty:
        return pd.DataFrame(columns=['candle_begin_time', 'symbol', 'target_alloc_ratio'])

    # 转换spot和swap的选币数据为透视表，以candle_begin_time为index，symbol为columns，values为target_alloc_ratio的sum
    # 注：多策略的相同周期的相同选币，会在这个步骤被聚合权重
    df_ratio = df_select.pivot(index='candle_begin_time', columns='symbol', values='target_alloc_ratio')

    # 构建candle_begin_time序列
    candle_begin_times = pd.date_range(
        df_select['candle_begin_time'].min(), df_select['candle_begin_time'].max(), freq='h', inclusive='both')
    df_ratio = df_ratio.reindex(candle_begin_times, fill_value=0)

    # 多offset的权重聚合
    df_ratio = df_ratio.rolling(stg_conf.hold_period, min_periods=1).sum()

    # 恢复 candle_begin_time, symbol, target_alloc_ratio的df结构
    df_ratio = df_ratio.stack().reset_index(name='target_alloc_ratio')
    df_ratio.rename(columns={'level_0': 'candle_begin_time'}, inplace=True)

    return df_ratio


def agg_multi_strategy_ratio(conf: BacktestConfig, df_select: pd.DataFrame):
    """
    聚合多offset、多策略选币结果中的target_alloc_ratio
    :param conf: 回测配置
    :param df_select: 选币结果
    :return: 聚合后的df_spot_ratio 和 df_swap_ratio。

    数据结构:
    - index_col为candle_begin_time，
    - columns为symbol，
    - values为target_alloc_ratio的聚合结果

    示例:
                    1000BONK-USDT	1000BTTC-USDT	1000FLOKI-USDT	1000LUNC-USDT	1000PEPE-USDT	1000RATS-USDT	1000SATS-USDT	1000SHIB-USDT	1000XEC-USDT	1INCH-USDT	AAVE-USDT	ACE-USDT	ADA-USDT	    AEVO-USDT   ...
    2021/1/1 00:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 01:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 02:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 03:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 04:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 05:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 06:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 07:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 08:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 09:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    """
    # ====================================================================================================
    # 1. 先针对每个策略的多offset进行聚合
    # ====================================================================================================
    df_spot_select_list = []
    df_swap_select_list = []

    # 如果是D的持仓周期，应该是当天的选币，第二天0点持仓。
    # 按照目前的逻辑，原来自带的begin time是0点
    if conf.is_day_period:
        df_select['candle_begin_time'] = df_select['candle_begin_time'] + pd.Timedelta(hours=23)

    for stg_conf in conf.strategy_list:
        # 裁切当前策略的spot选币结果
        df_select_spot = df_select[(df_select['strategy'] == stg_conf.name) & (df_select['is_spot'] == 1)]
        # 买入现货部分
        _spot_select_long = agg_offset_by_strategy(df_select_spot[df_select_spot['方向'] == 1], stg_conf)
        df_spot_select_list.append(_spot_select_long)
        # 做空现货部分
        _spot_select_short = agg_offset_by_strategy(df_select_spot[df_select_spot['方向'] == -1], stg_conf)
        df_spot_select_list.append(_spot_select_short)

        # 裁切当前策略的swap选币结果
        df_select_swap = df_select[(df_select['strategy'] == stg_conf.name) & (df_select['is_spot'] == 0)]
        # 买入合约部分
        _swap_select_long = agg_offset_by_strategy(df_select_swap[df_select_swap['方向'] == 1], stg_conf)
        df_swap_select_list.append(_swap_select_long)
        # 做空合约部分
        _swap_select_short = agg_offset_by_strategy(df_select_swap[df_select_swap['方向'] == -1], stg_conf)
        df_swap_select_list.append(_swap_select_short)

    df_spot_select = pd.concat(df_spot_select_list, ignore_index=True)
    df_swap_select = pd.concat(df_swap_select_list, ignore_index=True)

    # ====================================================================================================
    # 2. 针对多策略进行聚合
    # ====================================================================================================
    # 构建candle_begin_time序列，不管是D还是H的持仓周期，都以H为准
    candle_begin_times = pd.date_range(conf.start_date, conf.end_date, freq='h', inclusive='left')

    # 转换spot和swap的选币数据为透视表，以candle_begin_time为index，symbol为columns，values为target_alloc_ratio的sum
    # 注：多策略的相同周期的相同选币，会在这个步骤被聚合权重
    df_spot_ratio = to_ratio_pivot(df_spot_select, candle_begin_times, 'symbol')
    df_swap_ratio = to_ratio_pivot(df_swap_select, candle_begin_times, 'symbol')

    # # 针对下架币的处理
    # df_spot_ratio = trim_ratio_delists(df_spot_ratio, candle_begin_times.max(), spot_dict, 'spot')
    # df_swap_ratio = trim_ratio_delists(df_swap_ratio, candle_begin_times.max(), swap_dict, 'swap')

    return df_spot_ratio, df_swap_ratio
