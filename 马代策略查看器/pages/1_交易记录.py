import datetime

import numpy as np
import pandas as pd
import streamlit as st
import config
from mytools.models import BackTestingResults, plotly_transactions, OHLCData
from mytools.streamlist_tools import list_all_factors, rank_top_n_value, rank_top_n_pct, custom_fig_for_display, \
    list_all_strategy

st.set_page_config(page_title="交易记录", page_icon="📈", layout="wide")
strategy_name = config.backtest_name


def count_profit(series):
    # 定义区间
    bins = [-float('inf')] + \
           [round(x, 1) for x in np.arange(-0.5, 0.5 + 0.1, 0.1)] + \
           [float('inf')]

    # 统计每个区间的个数
    counts = pd.cut(series, bins=bins, right=False).value_counts().sort_index()

    # 创建结果字典
    result = {}

    # 小于-0.5的部分
    result['< -0.5'] = counts.iloc[0]

    # -0.5到0.5之间的区间
    for i in range(1, len(counts) - 1):
        interval = f'[{bins[i]}, {bins[i + 1]})'
        result[interval] = counts.iloc[i]

    # 大于0.5的部分
    result['> 0.5'] = counts.iloc[-1]

    return result


# 创建示例数据
@st.cache_data(show_spinner="加载交易记录中...")
def load_transaction(name: str = config.backtest_name):
    export_root = config.backtest_path / name
    transaction_root = export_root / "transactions"
    if not transaction_root.exists():
        transaction_root.mkdir(parents=True)
    cache_path = transaction_root / f"{name}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    results = BackTestingResults(name)
    trans_df = results.transaction_df(merge_trans=False)
    trans_df = trans_df[
        ['symbol', 'profit', 'start_time', 'end_time', 'start_close', 'end_close', 'direction', 'offset', 'is_spot']]
    trans_df = trans_df.sort_values(by='profit')
    trans_df.to_parquet(cache_path)
    return trans_df


@st.cache_data(max_entries=200, ttl=3600, persist=False, show_spinner="加载数据中...")
def cal_factor(name, p, symbol, use_spot) -> pd.Series:
    factor = f"{name}_{p}"
    return OHLCData().cal_factor_one_symbol(symbol, [(name, p)], use_spot=use_spot)[factor]


# 加载数据
# 在侧边栏显示 DataFrame
with st.sidebar:
    st.header(f"策略交易记录")
    options = list_all_strategy()
    strategy_name = st.selectbox("策略选择",
                                 options,
                                 index=options.index(strategy_name) if strategy_name in options else 0,
                                 key="strategy_options")
    transactions = load_transaction(strategy_name)
    date_range = st.date_input(
        "选择日期范围",
        value=(transactions['start_time'].min(), transactions['end_time'].max()),  # 默认范围为数据中的最小和最大日期
        min_value=transactions['start_time'].min(),
        max_value=transactions['end_time'].max()
    )

    if len(date_range) == 2:  # 确保选择了开始和结束日期
        start_date, end_date = date_range
        transactions = transactions[
            (transactions['start_time'] >= pd.to_datetime(start_date)) &
            (transactions['start_time'] <= pd.to_datetime(end_date))
            ]
    else:
        st.warning("请选择一个日期范围")

    plot_transactions = transactions[['symbol', 'profit', 'start_time', 'offset', 'direction']]
    # 使用 st.dataframe 显示可选择的表格
    st.write(f"总计{len(plot_transactions)}笔交易(请点击最左侧)")
    selected_indices = st.dataframe(
        plot_transactions,
        selection_mode="single-row",
        on_select="rerun",
        height=300,  # 设置高度避免太长
        key="sidebar_df",
        hide_index=True,
        column_config={
            "symbol": st.column_config.Column(width="small"),  # 25%
            "profit": st.column_config.Column(width="small"),  # 15%
            "start_time": st.column_config.Column(width="medium"),  # 20%
        }
    )

    long_position = transactions[transactions['direction'] == 1]
    short_position = transactions[transactions['direction'] == -1]
    long_profit_bin = count_profit(long_position['profit'])
    short_profit_bin = count_profit(short_position['profit'])
    st.write(f"收益分布")
    st.dataframe(pd.DataFrame({
        "做多": long_profit_bin,
        "做空": short_profit_bin,
    }))

if selected_indices.get('selection') and selected_indices['selection']['rows']:
    selected_row_index = selected_indices['selection']['rows'][0]
    selected_row = transactions.iloc[selected_row_index]

    direction_description = "做多" if selected_row['direction'] == 1 else "做空"
    # 在主页面显示选中的内容
    st.write("**选币内容：**")
    # 或者用其他方式显示
    st.write(
        f"币种: {selected_row['symbol']},盈亏: {selected_row['profit'] * 100:.2f}%,方向: {direction_description}, offset: {selected_row['offset']}, 时间: {selected_row['start_time']} to {selected_row['end_time']}")
else:
    st.write("请在侧边栏中选择一行数据")

# 有选择的数量
if selected_indices.get('selection') and selected_indices['selection']['rows']:
    selected_row_index = selected_indices['selection']['rows'][0]
    selected_row = transactions.iloc[selected_row_index]
    symbol = selected_row['symbol']
    results = BackTestingResults(strategy_name)
    stg_config = results.config
    use_spot = stg_config.is_use_spot
    start_time = selected_row['start_time']
    end_time = selected_row['end_time']
    factor_config = stg_config.strategy_list[0].factor_list[0]
    factor_name = factor_config[0]
    factor_parameter = factor_config[2]
    factor_ascending = factor_config[1]
    trade_data = OHLCData()
    data = trade_data.get_data_by_symbol(symbol, use_spot)
    data[factor_name] = cal_factor(factor_name, factor_parameter, symbol, use_spot)
    draw_cols = [factor_name]
    gap = end_time - start_time
    delta_hour = max(gap * 0.5, datetime.timedelta(hours=24 * 3))
    start_date = start_time - delta_hour
    end_date = end_time + delta_hour
    plot_df = data[(data['candle_begin_time'] >= start_date) & (data['candle_begin_time'] <= end_date)]

    back_hour = delta_hour.total_seconds() / 3600
    # 示例1：年龄的柱状图
    fig = plotly_transactions(plot_df, draw_cols, back_hour, start_time, end_time,
                              symbol, None, "持仓时间", "持仓", True)

    st.plotly_chart(custom_fig_for_display(fig))  # 不使用 use_container_width

    with (st.form(key="自定义因子")):
        st.subheader("实时计算因子")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            options = list_all_factors()
            new_factor = st.selectbox("因子",
                                      options,
                                      index=options.index(factor_name) if factor_name in options else 0,
                                      key="factor_select")
        with col2:
            new_parameter = st.number_input("参数", value=factor_parameter, key="factor_parameter")
        with col3:
            new_select_number = st.number_input("选择数量(0为不计算整体)", value=0.0, step=0.01, key="num2")
        with col4:
            new_ascending_str = st.selectbox("排序",
                                             ['升序', "降序"],
                                             key="new_ascending_str")
            new_ascending = True if new_ascending_str == "升序" else False

        submit_button = st.form_submit_button(label="计算新因子")

    if submit_button:
        with st.spinner("计算新因子，请稍后"):

            st.write(
                f"选项: {new_factor}, 参数: {new_parameter},选择数量: {new_select_number}, 排序: {new_ascending_str}，用现货:{use_spot}")
            if new_select_number == 0:
                factor_new_df = OHLCData().cal_factor_one_symbol(symbol, [(new_factor, new_parameter)], use_spot=use_spot)
                if factor_new_df.empty:
                    st.warning(f'{symbol} {"现货" if use_spot else "合约"} 数据不存在，跳过后续计算')
                else:
                    factor_new_df = factor_new_df[(factor_new_df['candle_begin_time'] >= start_date) & (
                            factor_new_df['candle_begin_time'] <= end_date)]
                    factor_new_col_name = f"{new_factor}_{new_parameter}"
                    new_factor_col = [factor_new_col_name]
                    fig = plotly_transactions(factor_new_df, new_factor_col, back_hour, start_time, end_time,
                                              symbol, None, "持仓时间", "持仓", True)
                    st.plotly_chart(custom_fig_for_display(fig))
            else:
                factor_new_df = OHLCData().calculate_factors([(new_factor, new_parameter)], use_spots=use_spot)

                # 新画的factor
                factor_new_symbol = factor_new_df.loc[factor_new_df['symbol'] == symbol]
                if factor_new_symbol.empty:
                    st.warning(f'{symbol} {"现货" if use_spot else "合约"} 数据不存在，跳过后续计算')
                else:
                    new_factor_plot_df = factor_new_symbol[(factor_new_symbol['candle_begin_time'] >= start_date) & (
                            factor_new_symbol['candle_begin_time'] <= end_date)]

                    factor_new_col_name = f"{new_factor}_{new_parameter}"
                    top_factor_new_col_name = f"{factor_new_col_name}_threshold"

                    all_new_factor_df = factor_new_df[(factor_new_df['candle_begin_time'] >= start_date) & (
                            factor_new_df['candle_begin_time'] <= end_date)].copy()

                    all_new_factor_df['rank'] = all_new_factor_df.groupby('candle_begin_time')[factor_new_col_name].rank(
                        method='min', ascending=new_ascending)
                    all_new_factor_df['总币数'] = all_new_factor_df.groupby('candle_begin_time')['symbol'].transform('size')
                    select_num = np.ceil(all_new_factor_df['总币数'] * new_select_number).tolist() if int(new_select_number) == 0 else new_select_number
                    all_new_factor_df['select_num'] = select_num
                    # all_new_factor_df.to_csv("all_new_factor_df.csv", index=False, encoding='utf-8-sig')
                    top_df = all_new_factor_df[all_new_factor_df['rank'] == select_num]
                    top_df = top_df[['candle_begin_time', factor_new_col_name]]
                    new_factor_col = [factor_new_col_name, top_factor_new_col_name]

                    new_factor_plot_df = pd.merge(new_factor_plot_df, top_df,
                                                  on='candle_begin_time', how='left')
                    new_factor_plot_df.rename(columns={
                        f'{factor_new_col_name}_x': factor_new_col_name,
                        f'{factor_new_col_name}_y': top_factor_new_col_name,

                    }, inplace=True)
                    # new_factor_plot_df.to_csv("new_factor_plot_df.csv", index=False, encoding='utf-8-sig')
                    fig = plotly_transactions(new_factor_plot_df, new_factor_col, back_hour, start_time, end_time,
                                              symbol, None, "持仓时间", "持仓", True)

                    st.plotly_chart(custom_fig_for_display(fig))
