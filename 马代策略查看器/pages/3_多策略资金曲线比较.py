import pandas as pd
import streamlit as st

import config
from core.evaluate import strategy_evaluate
from core.utils.path_kit import get_file_path
from mytools.constants import DEFAULT_PERIOD_CONFIG
from mytools.models import BackTestingResults
from mytools.streamlist_tools import list_all_strategy, custom_fig_for_display
from mytools.xbx import draw_equity_curve_plotly_xbx

st.set_page_config(page_title="多策略资金曲线比较", page_icon="📈", layout="wide")

st.header(f"多策略资金曲线比较")


options = list_all_strategy()
select_strategies = st.multiselect(
    "策略选择",
    options,
    placeholder="选择需要比较的策略",
    key="strategy_options"
)

long_short_select_strategy = st.selectbox(
    "显示多空比的策略",
    ['无', *select_strategies],
    index=0,
    key="long_short_select_type")

date_select_type = st.selectbox("日期方式",
                                ["自定义", '预设'],
                                key="date_select_type")

### 日期选择
if date_select_type == "预设":
    period_dict = dict()
    for period_1st in DEFAULT_PERIOD_CONFIG:
        sub_period = DEFAULT_PERIOD_CONFIG[period_1st]
        for period in sub_period:
            data_range = sub_period[period]
            period_start = data_range[0]
            period_end = data_range[1]
            period_dict[f"{period_1st}-{period}-{period_start}-{period_end}"] = (period_start, period_end)

    select_period = st.selectbox("周期选择",
                                 list(period_dict.keys()),
                                 key="select_period")
    select_start, select_end = period_dict[select_period]
else:
    date_range = st.date_input(
        "选择日期范围",
        value=(config.start_date, config.end_date),  # 默认范围为数据中的最小和最大日期
        min_value=config.start_date,
        max_value=config.end_date
    )
    try:
        select_start, select_end = date_range
    except ValueError:
        st.error("请完整选择两个日期（开始)")
        st.stop()  # 停止后续代码执行


submit_button = st.button(label="合并")

st.title("资金曲线")

if submit_button:

    if not select_strategies:
        st.warning('选择需要比较的策略')

    # 不显示子图
    show_subplots = False

    plot_data = pd.DataFrame(pd.date_range(start=select_start, end=select_end, freq='H'), columns=['candle_begin_time'])
    plot_data = plot_data.set_index('candle_begin_time')

    # 遍历设置的策略，然后进行策略评价
    for strategy in select_strategies:
        results = BackTestingResults(strategy)
        equity_df = results.equity
        equity_df = equity_df[
            (equity_df['candle_begin_time'] >= pd.to_datetime(select_start)) &
            (equity_df['candle_begin_time'] <= pd.to_datetime(select_end))
            ]

        if equity_df['candle_begin_time'].min() > pd.to_datetime(select_start):
            st.warning(f"{strategy} 资金曲线开始时间: {equity_df['candle_begin_time'].min()}, 数据长度不足")

        equity_df['净值'] = equity_df['equity'] / equity_df['equity'].iloc[0]
        equity_df['涨跌幅'] = equity_df['净值'].pct_change()
        rtn, _, _, _ = strategy_evaluate(equity_df, net_col='净值', pct_col='涨跌幅')

        equity_df = equity_df.set_index('candle_begin_time')
        plot_data[f'净值_{strategy}'] = equity_df['净值']
        plot_data[f'回撤_{strategy}'] = equity_df['净值dd2here']

        if strategy == long_short_select_strategy:
            show_subplots = True
            # 计算仓位比例
            plot_data['long_pos_ratio'] = equity_df['long_pos_value'] / equity_df['equity']
            plot_data['short_pos_ratio'] = equity_df['short_pos_value'] / equity_df['equity']
            plot_data['empty_ratio'] = (results.config.leverage - plot_data['long_pos_ratio'] - plot_data['short_pos_ratio']).clip(lower=0)
            # 计算累计值，主要用于后面画图使用
            plot_data['long_cum'] = plot_data['long_pos_ratio']
            plot_data['short_cum'] = plot_data['long_pos_ratio'] + plot_data['short_pos_ratio']
            plot_data['empty_cum'] = results.config.leverage  # 空仓占比始终为 1（顶部）
            # 多空选币数量
            plot_data['symbol_long_num'] = equity_df['symbol_long_num']
            plot_data['symbol_short_num'] = equity_df['symbol_short_num']

    all_swap = pd.read_pickle(config.swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=plot_data,
                          right=btc_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['BTC涨跌幅'] = account_df['close'].pct_change()
    account_df['BTC涨跌幅'].fillna(value=0, inplace=True)
    account_df['BTC资金曲线'] = (account_df['BTC涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['BTC涨跌幅']

    eth_df = all_swap['ETH-USDT']
    account_df = pd.merge(left=account_df,
                          right=eth_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['ETH涨跌幅'] = account_df['close'].pct_change()
    account_df['ETH涨跌幅'].fillna(value=0, inplace=True)
    account_df['ETH资金曲线'] = (account_df['ETH涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['ETH涨跌幅']

    # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
    data_dict = {}
    right_axis = {}
    for col in account_df.columns:
        if '净值' in col:
            data_dict[col] = col
        if '回撤' in col:
            right_axis[col] = col

    data_dict.update({'BTC资金曲线': 'BTC资金曲线', 'ETH资金曲线': 'ETH资金曲线'})

    # 如果画多头、空头资金曲线，同时也会画上回撤曲线
    pic_title = f"多策略资金曲线对比_{select_start}-{select_end}_{r'_'.join([_ for _ in select_strategies])}"
    desc = f'显示多空比的策略:{long_short_select_strategy}'
    # 调用画图函数
    fig = draw_equity_curve_plotly_xbx(account_df,
                                       data_dict=data_dict,
                                       date_col='candle_begin_time',
                                       right_axis=right_axis,
                                       title=pic_title,
                                       desc=desc,
                                       show_subplots=show_subplots)
    fig = custom_fig_for_display(fig)
    save_file_path = get_file_path(config.backtest_path.parent, '多策略资金曲线比较', f'{pic_title}.html')
    fig.write_html(save_file_path)
    st.success(f'文件已经保存到：{save_file_path}')
    st.plotly_chart(fig)
