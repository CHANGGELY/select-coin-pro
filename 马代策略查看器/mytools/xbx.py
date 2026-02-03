import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import swap_path
from core.evaluate import strategy_evaluate
from core.model.backtest_config import BacktestConfig
import streamlit as st


def draw_equity_curve_plotly_xbx(df, data_dict, date_col=None, right_axis=None, pic_size=None, chg=False, title=None,
                                 desc=None, show_subplots=False, right_is_line: bool = False):
    """
    绘制策略曲线
    :param right_is_line:
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param title: 标题
    :param desc: 图片描述
    :param show_subplots: 是否展示子图，显示多空仓位比例
    :return:
    """
    if pic_size is None:
        pic_size = [1500, 920]

    draw_df = df.copy()

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 创建子图
    row = 3 if show_subplots else 1
    row_heights = [0.8, 0.1, 0.1] if show_subplots else [1]
    specs = [[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]] if show_subplots else [[{"secondary_y": True}]]
    fig = make_subplots(
        rows=row, cols=1,
        shared_xaxes=True,  # 共享 x 轴，主，子图共同变化
        vertical_spacing=0.03,  # 减少主图和子图之间的间距
        row_heights=row_heights,  # 主图高度占 80%，子图高度占 10%
        specs=specs
    )

    # 主图：绘制左轴数据
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[data_dict[key]], name=key), row=1, col=1)

    # 绘制右轴数据
    if right_axis:
        key = list(right_axis.keys())[0]
        if right_is_line is True:
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
        else:
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                     # marker_color='orange',
                                     opacity=0.1,
                                     line=dict(width=0),
                                     fill='tozeroy',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
        for i, key in enumerate(list(right_axis.keys())[1:]):
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     marker=dict(color=f'rgba({100 + i * 50}, {100 + i * 20}, {100 + i * 30}, 0.8)'),
                                     opacity=0.2, line=dict(width=0),
                                     fill='tozeroy',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴

    if show_subplots:
        # 子图：按照 matplotlib stackplot 风格实现堆叠图
        # 最下面是多头仓位占比
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['long_cum'],
            mode='lines',
            line=dict(width=0),
            fill='tozeroy',
            fillcolor='rgba(30, 177, 0, 0.6)',
            name='多头仓位占比',
            hovertemplate="多头仓位占比: %{customdata:.4f}<extra></extra>",
            customdata=draw_df['long_pos_ratio']  # 使用原始比例值
        ), row=2, col=1)

        # 中间是空头仓位占比
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['short_cum'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 99, 77, 0.6)',
            name='空头仓位占比',
            hovertemplate="空头仓位占比: %{customdata:.4f}<extra></extra>",
            customdata=draw_df['short_pos_ratio']  # 使用原始比例值
        ), row=2, col=1)

        # 最上面是空仓占比
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['empty_cum'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 46, 77, 0.6)',
            name='空仓占比',
            hovertemplate="空仓占比: %{customdata:.4f}<extra></extra>",
            customdata=draw_df['empty_ratio']  # 使用原始比例值
        ), row=2, col=1)

        # 子图：右轴绘制 long_short_ratio 曲线
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['symbol_long_num'],
            name='多头选币数量',
            mode='lines',
            line=dict(color='rgba(30, 177, 0, 0.6)', width=2)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['symbol_short_num'],
            name='空头选币数量',
            mode='lines',
            line=dict(color='rgba(255, 99, 77, 0.6)', width=2)
        ), row=3, col=1)

    fig.update_layout(template="none", width=pic_size[0], height=pic_size[1], title_text=title,
                      hovermode="x unified", hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)', ),
                      annotations=[
                          dict(
                              text=desc,
                              xref='paper',
                              yref='paper',
                              x=0.5,
                              y=1.05,
                              showarrow=False,
                              font=dict(size=12, color='black'),
                              align='center',
                              bgcolor='rgba(255,255,255,0.8)',
                          )
                      ]
                      )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="线性 y轴",
                         method="relayout",
                         args=[{"yaxis.type": "linear"}]),
                    dict(label="Log y轴",
                         method="relayout",
                         args=[{"yaxis.type": "log"}]),
                ])],
    )

    fig.update_yaxes(
        showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    fig.update_xaxes(
        showspikes=True, spikemode='across+marker', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    return fig


def show_plot_performance_with_factor(conf: BacktestConfig, account_df, rtn, factor_colum: str, title_prefix='',
                                      debug: bool = False, **kwargs, ):
    account_df['long_pos_ratio'] = account_df['long_pos_value'] / account_df['equity']
    account_df['short_pos_ratio'] = account_df['short_pos_value'] / account_df['equity']
    account_df['empty_ratio'] = (conf.leverage - account_df['long_pos_ratio'] - account_df['short_pos_ratio']).clip(
        lower=0)
    # 计算累计值，主要用于后面画图使用
    account_df['long_cum'] = account_df['long_pos_ratio']
    account_df['short_cum'] = account_df['long_pos_ratio'] + account_df['short_pos_ratio']
    account_df['empty_cum'] = conf.leverage  # 空仓占比始终为 1（顶部）

    all_swap = pd.read_pickle(swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=account_df,
                          right=btc_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['BTC涨跌幅'] = account_df['close'].pct_change()
    account_df['BTC涨跌幅'].fillna(value=0, inplace=True)
    account_df['BTC资金曲线'] = (account_df['BTC涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['BTC涨跌幅']

    account_df['涨跌幅'].fillna(value=0, inplace=True)
    account_df['净值'] = (account_df['涨跌幅'] + 1).cumprod()
    rtn, _, _, _ = strategy_evaluate(account_df, net_col='净值', pct_col='涨跌幅')

    data_dict = {'多空资金曲线': '净值'}
    data_dict.update({'BTC资金曲线': 'BTC资金曲线'})
    right_axis = {'因子值': factor_colum}

    # 如果画多头、空头资金曲线，同时也会画上回撤曲线
    pic_title = f"{title_prefix}CumNetVal:{rtn.at['累积净值', 0]}, Annual:{rtn.at['年化收益', 0]}, MaxDrawdown:{rtn.at['最大回撤', 0]}"
    pic_desc = conf.get_fullname()
    if debug:
        st.dataframe(account_df)
    # 调用画图函数
    return draw_equity_curve_plotly_xbx(account_df,
                                        data_dict=data_dict,
                                        date_col='candle_begin_time',
                                        right_axis=right_axis,
                                        title=pic_title,
                                        desc=pic_desc,
                                        pic_size=[1000, 600],
                                        show_subplots=True,
                                        right_is_line=True)


def show_plot_performance(conf: BacktestConfig, account_df, rtn=None, title_prefix='', debug: bool = False, **kwargs):
    # 计算仓位比例
    account_df['long_pos_ratio'] = account_df['long_pos_value'] / account_df['equity']
    account_df['short_pos_ratio'] = account_df['short_pos_value'] / account_df['equity']
    account_df['empty_ratio'] = (conf.leverage - account_df['long_pos_ratio'] - account_df['short_pos_ratio']).clip(
        lower=0)
    # 计算累计值，主要用于后面画图使用
    account_df['long_cum'] = account_df['long_pos_ratio']
    account_df['short_cum'] = account_df['long_pos_ratio'] + account_df['short_pos_ratio']
    account_df['empty_cum'] = conf.leverage  # 空仓占比始终为 1（顶部）

    all_swap = pd.read_pickle(swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=account_df,
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

    if rtn is None:
        account_df['涨跌幅'].fillna(value=0, inplace=True)
        account_df['净值'] = (account_df['涨跌幅'] + 1).cumprod()
        rtn, _, _, _ = strategy_evaluate(account_df, net_col='净值', pct_col='涨跌幅')

    # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
    data_dict = {'多空资金曲线': '净值'}
    for col_name, col_series in kwargs.items():
        account_df[col_name] = col_series
        data_dict[col_name] = col_name
    data_dict.update({'BTC资金曲线': 'BTC资金曲线', 'ETH资金曲线': 'ETH资金曲线'})
    right_axis = {'多空最大回撤': '净值dd2here'}

    # 如果画多头、空头资金曲线，同时也会画上回撤曲线
    pic_title = f"{title_prefix}CumNetVal:{rtn.at['累积净值', 0]}, Annual:{rtn.at['年化收益', 0]}, MaxDrawdown:{rtn.at['最大回撤', 0]}"
    pic_desc = conf.get_fullname()
    if debug:
        st.dataframe(account_df)
    # 调用画图函数
    return draw_equity_curve_plotly_xbx(account_df,
                                        data_dict=data_dict,
                                        date_col='candle_begin_time',
                                        right_axis=right_axis,
                                        title=pic_title,
                                        desc=pic_desc,
                                        show_subplots=True)
