import numpy as np
import pandas as pd
import streamlit as st

import config
from core.model.strategy_config import filter_common, FilterFactorConfig, FilterMethod
from mytools.constants import DEFAULT_PERIOD_CONFIG
from mytools.models import BackTestingResults, OHLCData
from mytools.streamlist_tools import list_all_strategy, custom_fig_for_display, list_all_factors
from mytools.xbx import show_plot_performance, show_plot_performance_with_factor

st.set_page_config(page_title="资金曲线", page_icon="📈", layout="wide")
strategy_name = config.backtest_name

with st.sidebar:
    st.header(f"策略资金曲线查看")
    avail_factors = list_all_strategy()
    select_strategy = st.selectbox("策略选择",
                                   avail_factors,
                                   index=avail_factors.index(strategy_name) if strategy_name in avail_factors else 0,
                                   key="strategy_options")

date_select_type = st.selectbox("日期方式",
                                ["自定义", '预设'],
                                key="date_select_type")

st.title("资金曲线")

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

results = BackTestingResults(select_strategy)
equity_df = results.equity
plot_df = equity_df[
    (equity_df['candle_begin_time'] >= pd.to_datetime(select_start)) &
    (equity_df['candle_begin_time'] <= pd.to_datetime(select_end))
    ]
bt_config = results.config
fig = show_plot_performance(bt_config, plot_df, debug=False)
st.plotly_chart(custom_fig_for_display(fig))

# 以下内容暂时不开放
# with st.form(key="自定义因子"):
#     st.subheader("自定义参数")
#     factor_config = config.strategy_list[0]['factor_list'][0]
#     factor_parameter = factor_config[2]
#     factor_name = factor_config[0]
#     factor_ascending = factor_config[1]
#     avail_factors = list_all_factors()
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         new_factor = st.selectbox("因子",
#                                   avail_factors,
#                                   index=avail_factors.index(factor_name) if factor_name in avail_factors else 0,
#                                   key="factor_select")
#     with col2:
#         new_parameter = st.number_input("参数", value=factor_parameter, key="factor_parameter")
#     with col3:
#         new_select_number = st.number_input("选择数量，0表示全部", value=0, key="num2")
#     with col4:
#         factor_ascending = st.selectbox("排序", ["升序", "降序"], key="factor_ascending")
#     with col5:
#         agg_method = st.selectbox("统计方式", ["平均", '求和', "最大", "最小"], key="agg_method")
#
#     st.subheader("过滤参数")
#
#     filter_enable_str = st.selectbox("启用过滤", ["关闭", "启用"], key="filter_enable_str")
#     filter_enable = True if filter_enable_str == "启用" else False
#
#     col1, col2, col3, col4, col5, col6 = st.columns(6)
#     with col1:
#         filter_factor_select = st.selectbox("因子",
#                                             avail_factors,
#                                             key="filter_factor_select")
#     with col2:
#         filter_parameter = st.number_input("参数", value=factor_parameter, key="filter_parameter")
#     with col3:
#         type_option = ['pct', 'rank', 'val']
#         filter_type = st.selectbox("类型", type_option, key="filter_type")
#     with col4:
#         filter_cal_option = ['>', '>=', '<', '<=', '==', '!=']
#         filter_cal = st.selectbox("计算式", filter_cal_option, key="filter_cal")
#     with col5:
#         filter_val = st.number_input('数值', value=0.1, step=0.01, key="filter_val")
#     with col6:
#         filter_ascending_str = st.selectbox("排序", ["升序", "降序"], key="filter_ascending_str")
#         filter_ascending = True if filter_ascending_str == "升序" else False
#
#     submit_button = st.form_submit_button(label="计算")
#
# if submit_button:
#     with (st.spinner("计算新因子，请稍后")):
#         select_ascending = False if factor_ascending == "从小到大" else True
#         st.write(
#             f"选项: {new_factor}, 参数: {new_parameter},选择数量: {new_select_number}, 统计方式: {agg_method}")
#
#         cal_factor_list = [(new_factor, new_parameter)]
#         if filter_enable:
#             cal_factor_list.append((filter_factor_select, filter_parameter))
#
#         factor_new_df = OHLCData().calculate_factors(cal_factor_list)
#         factor_new_col_name = f"{new_factor}_{new_parameter}"
#
#         all_new_factor_df = factor_new_df[(factor_new_df['candle_begin_time'] >= pd.to_datetime(select_start)) & (
#                 factor_new_df['candle_begin_time'] <= pd.to_datetime(select_end))]
#         if filter_enable:
#             filter_config = [
#                 FilterFactorConfig(
#                     name=filter_factor_select,
#                     param=filter_parameter,
#                     method=FilterMethod(
#                         how=filter_type,
#                         range=f"{filter_cal}{filter_val}"
#                     ),
#                     is_sort_asc=filter_ascending
#                 )
#
#             ]
#             filter_condition = filter_common(all_new_factor_df, filter_config)
#             all_new_factor_df = all_new_factor_df[filter_condition]
#         if new_select_number == 0:
#             top_df = all_new_factor_df
#         else:
#             all_new_factor_df['rank'] = all_new_factor_df.groupby('candle_begin_time')[factor_new_col_name].rank(
#                 method='min', ascending=factor_ascending)
#             all_new_factor_df['总币数'] = all_new_factor_df.groupby('candle_begin_time')['symbol'].transform('size')
#             select_num = np.ceil(all_new_factor_df['总币数'] * new_select_number).tolist() if int(new_select_number) == 0 else new_select_number
#             all_new_factor_df['select_num'] = select_num
#             top_df = all_new_factor_df[all_new_factor_df['rank'] == select_num]
#
#         if agg_method == "最大":
#             factor_data = top_df.groupby('candle_begin_time')[factor_new_col_name].max()
#         elif agg_method == "最小":
#             factor_data = top_df.groupby('candle_begin_time')[factor_new_col_name].min()
#         elif agg_method == "求和":
#             factor_data = top_df.groupby('candle_begin_time')[factor_new_col_name].sum()
#         else:
#             factor_data = top_df.groupby('candle_begin_time')[factor_new_col_name].mean()
#
#         plot_with_factor = pd.merge(plot_df, factor_data, on='candle_begin_time', how='left')
#         fig = show_plot_performance_with_factor(bt_config, plot_with_factor, factor_new_col_name, debug=False)
#         st.plotly_chart(custom_fig_for_display(fig))
