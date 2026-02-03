import asyncio
import collections
import concurrent.futures
import dataclasses
import datetime
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from tqdm import tqdm

import config
from core.backtest import step2_load_data
from core.model.backtest_config import BacktestConfig
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import get_logger
from core.utils.path_kit import get_folder_path

logger = get_logger()


class CAPData:
    _instance = None
    _lock = threading.Lock()

    HEAD_COLUMN = [
        "candle_begin_time",
        "symbol",
        "id",
        "name",
        "date_added",
        "max_supply",
        "circulating_supply",
        "total_supply",
        "usd_price",
        "max_mcap",  # 理论总市值
        "circulating_mcap",  # 流通市值
        "total_mcap"  # 实际总市值
    ]

    def __new__(cls, source_path: Path = None):
        if source_path is None:
            source_path = Path(config.data_source_dict['coin-cap'][1])

        if cls._instance is None:
            with cls._lock:  # 确保线程安全
                if cls._instance is None:  # 再次检查
                    cls._instance = super(CAPData, cls).__new__(cls)
                    cls._instance.initialize(source_path)  # 初始化数据
        return cls._instance

    def get_data_by_symbol(self, symbol: str) -> pd.DataFrame:
        return self.data[self.data.symbol == symbol]

    def initialize(self, source_path: Path = None):
        # 这里进行数据的初始化
        cache = source_path / "cache.parquet"
        if cache.exists():
            logger.info(f"loading cap data  from {cache}")
            self.data = pd.read_parquet(cache)
            return

        csv_files = source_path.glob('*.csv')
        # 创建事件循环
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cap_data = []
        job_num = max(os.cpu_count() - 1, 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=job_num) as executor:
            futures = [executor.submit(_read_one_cap, csv)
                       for csv in csv_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                cap_data.append(future.result())
        res: pd.DataFrame = pd.concat(cap_data, ignore_index=True)

        time_group = res.groupby('candle_begin_time')
        res['circulating_mcap_rank'] = time_group['circulating_mcap'].rank(method='min')
        res['total_mcap_rank'] = time_group['total_mcap'].rank(method='min')

        self.data = res

        self.data.to_parquet(cache)
        loop.close()


def _read_one_cap(csv_path: Path):
    # 模拟一个异步任务，随机等待一段时间
    csv = pd.read_csv(csv_path, parse_dates=["candle_begin_time"], encoding="GBK", skiprows=1)
    expanded_rows = []
    for index, row in csv.iterrows():
        date = row['candle_begin_time']
        # 获取当前行的所有值
        values = row.to_dict()  # 将行转换为字典
        for hour in range(24):
            # 创建一个新的字典，替换日期为当前小时
            new_row = values.copy()  # 复制当前行的所有值
            new_row['candle_begin_time'] = date + pd.Timedelta(hours=hour)  # 替换日期为小时
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)  # 返回结果


@dataclasses.dataclass
class AdditionInfo:
    hold_hour: int
    shift_num: int = None
    shift_col_name: str = None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['symbol'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class OHLCData:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):

        if cls._instance is None:
            with cls._lock:  # 确保线程安全
                if cls._instance is None:  # 再次检查
                    cls._instance = super(OHLCData, cls).__new__(cls)
                    cls._instance.initialize()  # 初始化数据
        return cls._instance

    def initialize(self, use_spot: bool = True):
        if hasattr(self, '_use_spot') and self._use_spot == use_spot:
            return
        logger.info("refresh ohlc data with spot:{}".format(use_spot))
        self._use_spot = use_spot
        my_config = BacktestConfig("test")
        my_config.is_use_spot = True  # 强制设置为 True，加载所有数据
        step2_load_data(my_config)
        cache_path = Path(get_folder_path('data', 'cache')) / "all_candle_df_list.pkl"
        data = pd.read_pickle(cache_path)
        self._data = {f"{i['symbol'].iloc[-1]}_{i['is_spot'].iloc[-1]}": clean_data(i) for i in data}

    def symbols(self, with_spots=True) -> Set[str]:
        swap = pd.read_pickle(config.spot_path)
        swaps_symbols = set(swap.keys())
        if with_spots:
            logger.debug("开始使用现货")
            spot = pd.read_pickle(config.swap_path)
            spots_symbols = set(spot.keys())
            return spots_symbols.union(swaps_symbols)
        return swaps_symbols

    def calculate_factors(self, factors: List[tuple[str, int]], hold_info: AdditionInfo = None,
                          use_spots: bool = True) -> pd.DataFrame:

        self.initialize(use_spots)
        symbols = self._data.keys()
        symbols = [_.replace('_0', '').replace('_1', '') for _ in symbols if _.endswith(f'_{1 if use_spots else 0}')]

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.job_num) as executor:
            futures = [executor.submit(self.cal_factor_one_symbol, symbol, factors, hold_info, use_spots)
                       for
                       symbol in symbols]
        res = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if not result.empty:
                res.append(result)
        return pd.concat(res, ignore_index=True)

    def cal_factor_one_symbol(self, symbol: str, factors: List[tuple[str, int]],
                              hold_info: AdditionInfo = None, use_spot: bool = True) -> pd.DataFrame:
        try:
            data = self.get_data_by_symbol(symbol, use_spot)
        except ValueError as e:
            print(f'{symbol} {"现货" if use_spot else "合约"} 数据不存在')
            return pd.DataFrame()

        for factor, hold_period in factors:
            factor_name = f"{factor}_{hold_period}"
            factor = FactorHub.get_by_name(factor)
            if hasattr(factor, 'extra_data_dict') and factor.extra_data_dict:
                from core.utils.functions import merge_data
                for data_name in factor.extra_data_dict.keys():
                    extra_data_dict = merge_data(data, data_name, factor.extra_data_dict[data_name])
                    for extra_data_name, extra_data_series in extra_data_dict.items():
                        data[extra_data_name] = extra_data_series.values
            data = factor.signal(data, hold_period, factor_name)

        if hold_info is not None:
            """
            因为原始数据从0点开始。所以比较方便做这方面的事情
            """
            hold_hour = hold_info.hold_hour
            res = []

            df = data.copy()
            # 这里算 offset 不够严谨，目前的两个页面走不到这个逻辑，暂不处理
            df['offset'] = data.index % hold_hour

            if hold_info.shift_col_name is not None:
                df[hold_info.shift_col_name] = df['close'].shift(hold_info.shift_num)
            df.dropna(subset=["symbol"], inplace=True)
            res.append(df)

            data = df

            data.sort_values(by='candle_begin_time', inplace=True)
            return data

        return data

    def get_data_by_symbol(self, symbol: str, _: bool = True) -> pd.DataFrame:
        """
        默认首先取swap的数据，然后再取spot的数据
        :param _:
        :param symbol:
        :return:
        """
        symbol_name = f"{symbol}_{1 if _ else 0}"
        if symbol_name in self._data:
            return self._data[symbol_name].copy()

        raise ValueError(f"{symbol_name}不存在")

    def all_in_one(self) -> pd.DataFrame:
        return pd.concat(self._data.values(), ignore_index=True)

    def refresh_close_and_profit(self, trans: pd.DataFrame) -> pd.DataFrame:
        all_in_one = self.all_in_one()[["symbol", "candle_begin_time", "is_spot", "close"]]
        trans = pd.merge(trans, all_in_one, how='left',
                         left_on=["symbol", "is_spot", "start_time"],
                         right_on=["symbol", "is_spot", "candle_begin_time"])
        trans = pd.merge(trans, all_in_one, how='left',
                         left_on=["symbol", "is_spot", "end_time"],
                         right_on=["symbol", "is_spot", "candle_begin_time"],
                         suffixes=('_start', '_end'))

        trans = trans[
            ["symbol", "start_time", "end_time", "close_start", "close_end", "direction", "offset", "is_spot"]]

        trans = trans.rename(
            columns={"close_start": "start_close",
                     "close_end": "end_close", }
        )
        trans['profit'] = (trans['end_close'] / trans['start_close'] - 1) * trans['direction']
        return trans


@dataclasses.dataclass
class AnalysisTransactions:
    symbol: str
    start_date: datetime.datetime
    start_close: float
    is_spot: int
    direction: int
    offset: Optional[int] = 0
    end_date: Optional[datetime.datetime] = None
    end_close: Optional[float] = None

    @property
    def profit(self) -> float:
        return (self.end_close / self.start_close - 1) * self.direction


class BackTestingResults:

    def __init__(self, name: str = None, root: Path = config.backtest_path):
        if name is None:
            name = config.backtest_name
        self._bt_name = name
        self._root = root / name
        self._cache = root / "cache"
        self.transactions_root = self._root / "transactions"

    @property
    def export_folder(self):
        return self._root

    @property
    def select_coins(self) -> pd.DataFrame:
        select_coin_path = self._root / "选币结果.pkl"
        return pd.read_pickle(select_coin_path)

    @property
    def rtn(self) -> pd.DataFrame:
        rtn_path = self._root / "策略评价.csv"
        csv = pd.read_csv(rtn_path, encoding="utf-8-sig", index_col=0)
        csv.rename(columns={'0': 0}, inplace=True)
        return csv

    @property
    def config(self) -> BacktestConfig:
        select_coin_path = self._root / "config.pkl"
        return pd.read_pickle(select_coin_path)

    @property
    def equity(self) -> pd.DataFrame:
        return pd.read_csv(self._root / "资金曲线.csv", parse_dates=["candle_begin_time"], encoding="utf-8-sig",
                           index_col=0)

    @lru_cache(maxsize=2)
    def list_transactions(self, merge_trans: bool = False) -> List[AnalysisTransactions]:
        """

        :param merge_trans: 是否要把transaction做合并。
        :return:
        """
        stg_list = self.config.strategy_list
        select_coins = self.select_coins

        res = []

        for stg in stg_list:
            cache_dict = collections.defaultdict(list)
            hold_hour_str = stg.hold_period
            hold_time = int(hold_hour_str[:-1])
            hold_time_type = hold_hour_str[-1]
            if hold_time_type.upper() == "H":
                delta_time = datetime.timedelta(hours=hold_time)
            else:
                delta_time = datetime.timedelta(days=hold_time)
            stg_select_coins = select_coins.loc[select_coins["strategy"] == stg.name]
            stg_coins_groups = stg_select_coins.groupby(['symbol', 'offset'], observed=False)

            for (symbol, offset), group in tqdm(stg_coins_groups, desc="分析transactions"):
                transactions = to_transactions(group, delta_time, symbol)
                cache_dict[symbol].extend(transactions)
            if merge_trans:
                results = Parallel(n_jobs=config.job_num)(
                    delayed(merge_transactions)(l, hold_time) for _, l in
                    tqdm(cache_dict.items(), desc="merge transactions"))
                res.extend([item for sublist in results for item in sublist])
            else:
                res.extend([item for sublist in cache_dict.values() for item in sublist])

        return res

    def transaction_df(self, merge_trans: bool = True) -> pd.DataFrame:
        transactions = self.list_transactions(merge_trans=merge_trans)
        data = {
            "symbol": [t.symbol for t in transactions],
            "start_time": [t.start_date for t in transactions],
            "end_time": [t.end_date for t in transactions],
            "start_close": [t.start_close for t in transactions],
            "end_close": [t.end_close for t in transactions],
            "direction": [t.direction for t in transactions],
            "offset": [t.offset for t in transactions],
            "is_spot": [t.is_spot for t in transactions],
            "profit": [t.profit for t in transactions],
        }

        res = pd.DataFrame(data)
        res = OHLCData().refresh_close_and_profit(res)
        # 参数覆盖会出现重复的记录
        res = res.drop_duplicates(subset=['symbol', 'start_time', 'end_time', 'offset', 'profit'])
        res.sort_values("profit", inplace=True, ascending=False)
        return res


def merge_transactions(transactions: List[AnalysisTransactions], hold_hours: int, merge_type: str = "max") -> List[
    AnalysisTransactions]:
    """
    因为在实际的transaction中，相临近的offset可能有相同的，但是在分析的时候，就会相同的的transactions
    输入的transactions需要相同的symbol
    :param transactions: 需要merge的transactions
    :param hold_hours: 持仓时间
    :param merge_type: merge的模式，我现在想到有遗下集中。现在只是实现max模式
        max:  连着的 transaction取收益最大的一个。如果相同，取第一个
        long：最长,  在transaction中，取持续时间最大的一个。如果相同，取第一个
        combine：start和end都取最大的一个。
    :return:
    """
    if len(transactions) <= 1:
        return transactions

    sorted_transaction = sorted(transactions, key=lambda trans: trans.start_date)
    res = []
    prev = sorted_transaction[0]
    ready_to_merge = [prev]
    cache = sorted_transaction[1:]
    for t in cache:
        gap_hour = int((t.start_date - prev.start_date).total_seconds() / 3600)
        # (next_number - current) % (end - start + 1)  这个公式的简化。
        gap_offset = (t.offset - prev.offset) % hold_hours
        if gap_offset == gap_hour and t.direction == prev.direction:
            """
            因为计算出来的transaction,相同的offset，在下一个周期会做合并。所以这里没有考虑这里做特殊处理。
            但是可能有一些情况
            比如一个3H。
            第一个周期的offset，o1,o2,o3都有
            第二个周期的offset, 空，o2,o3
            那么按道理里说。这里应该会成为一个transaction。
            但是这里，会成merge成两个。太复杂了。不想考虑了。
            """
            ready_to_merge.append(t)
        else:
            max_t = max(ready_to_merge, key=lambda tran: tran.profit)
            res.append(max_t)
            ready_to_merge = [t]
        prev = t
    if len(ready_to_merge) > 0:
        max_t = max(ready_to_merge, key=lambda tran: tran.profit)
        res.append(max_t)
    return res


def to_transactions(df: pd.DataFrame, hold_period: datetime.timedelta, symbol: str) -> List[AnalysisTransactions]:
    res = []
    curr_transactions = None
    prev_time = None
    for index, row in df.iterrows():
        cbt = row['candle_begin_time']
        close = row['close']
        next_close = 0
        direction = row['方向']
        offset = row['offset']
        is_spot = row['is_spot']
        next_date = cbt + hold_period

        if curr_transactions is None:
            curr_transactions = _new_transaction(cbt, close, next_date, next_close, direction, is_spot, symbol, offset)
            prev_time = cbt
            continue

        if direction != curr_transactions.direction:
            res.append(curr_transactions)
            curr_transactions = _new_transaction(cbt, close, next_date, next_close, direction, is_spot, symbol, offset)
            continue

        if (cbt - prev_time) > hold_period:
            res.append(curr_transactions)
            curr_transactions = _new_transaction(cbt, close, next_date, next_close, direction, is_spot, symbol, offset)
        else:
            curr_transactions.end_close = close
            curr_transactions.end_date = next_date
        prev_time = cbt

    res.append(curr_transactions)

    return res


def _new_transaction(start_date, start_close, end_date, end_close, direction, is_spot, symbol, offset):
    curr_transactions = AnalysisTransactions(
        symbol=symbol,
        start_date=start_date,
        start_close=start_close,
        end_date=end_date,
        end_close=end_close,
        direction=direction,
        is_spot=is_spot,
        offset=offset
    )
    return curr_transactions

def plotly_transactions(df, show_list, back_hour, enter_time, exit_time, symbol,
                        mark_point_list=None, hold_text=None, title=None, return_fig=False,
                        addition_factor: List[str] = None):
    has_addition_factor = False
    other_factors = set()
    other_factors_length = 0
    if addition_factor is not None:
        other_factors = set(addition_factor) & set(df.columns)
        other_factors_length = len(other_factors)
        if other_factors_length > 0:
            has_addition_factor = True

        # 创建带有双Y轴的图表
    if has_addition_factor:
        rows = 2 + other_factors_length
        specs = [[{"secondary_y": True}], [{}]] + [[{}]] * other_factors_length
        subplot_titles = [title, None] + [None] * other_factors_length
        row_height = [4, 1] + [1] * other_factors_length
        fig = make_subplots(rows=rows,
                            cols=1,
                            shared_xaxes=True,
                            # vertical_spacing=0.01,
                            vertical_spacing=0.0,
                            specs=specs,  # 添加次级Y轴
                            subplot_titles=subplot_titles,
                            # row_heights=[4, 3])
                            row_heights=row_height)  # 修改两个区域的占比
    else:
        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            # vertical_spacing=0.01,
                            vertical_spacing=0.0,
                            specs=[[{"secondary_y": True}], [{}]],  # 添加次级Y轴
                            subplot_titles=[title, None],
                            # row_heights=[4, 3])
                            row_heights=[4, 1])

        # 添加K线图
    fig.add_trace(
        go.Candlestick(x=df['candle_begin_time'],
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'],
                       name="K线"),
        secondary_y=False,
    )

    if has_addition_factor:
        row_start = 3
        for factor_name in other_factors:
            fig.add_trace(
                go.Scatter(x=df['candle_begin_time'], y=df[factor_name], name=factor_name, line=dict(width=2)),
                row=row_start, col=1)
            row_start = row_start + 1

    # 根据show_list添加显示的线
    for col in show_list:
        fig.add_trace(
            go.Scatter(x=df['candle_begin_time'], y=df[col], name=col, line=dict(width=2)),
            secondary_y=True,
        )

    # 添加成交额图
    volume_color = ['#3D9970' if close_price >= open_price else '#FF4136' for open_price, close_price in
                    zip(df['open'], df['close'])]
    volume_bar = go.Bar(x=df['candle_begin_time'],
                        y=df['quote_volume'],
                        name='成交额',
                        marker_color=volume_color
                        )
    fig.add_trace(volume_bar, row=2, col=1)

    # 设置图表标题
    # fig.update_layout(title=f'{symbol}'"走势线图和"f'{factor}', xaxis_rangeslider_visible=False)
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(t=40, r=0, b=0))

    # 设置Y轴标题
    fig.update_yaxes(title_text="K线价格", secondary_y=False)
    fig.update_yaxes(title_text="成交额", row=2, col=1)
    fig.update_yaxes(title_text=f'因子值', secondary_y=True)
    if has_addition_factor:
        row_num = 3
        for factor_name in other_factors:
            fig.update_yaxes(title_text=factor_name, row=row_num, col=1)
            row_num = row_num + 1

    if hold_text is not None:
        # 添加垂直虚线和背景色
        fig.update_layout(shapes=[
            dict(type="line", x0=enter_time, y0=0, x1=enter_time, y1=1, xref='x', yref='paper',
                 line=dict(color="green", width=2, dash="dash")),
            dict(type="line", x0=exit_time, y0=0, x1=exit_time, y1=1, xref='x', yref='paper',
                 line=dict(color="blue", width=2, dash="dash")),
            dict(type="line", x0=enter_time - pd.Timedelta(hours=back_hour), y0=0,
                 x1=enter_time - pd.Timedelta(hours=back_hour), y1=1, xref='x', yref='paper',
                 line=dict(color="red", width=2, dash="dash")),
            dict(type="rect", x0=enter_time, y0=0, x1=exit_time, y1=1, xref='x', yref='paper', fillcolor="#D5E1DF",
                 opacity=1.0, layer="below", line_width=0),
            dict(type="rect", x0=enter_time - pd.Timedelta(hours=back_hour), y0=0, x1=enter_time, y1=1, xref='x',
                 yref='paper', fillcolor="#E6DAE0", opacity=1.0, layer="below", line_width=0),
        ])

        # 在虚线旁边添加文字
        enter_x = enter_time - pd.Timedelta(hours=round(back_hour / 2))
        hold_x = enter_time + (exit_time - enter_time) / 2
        exit_x = exit_time + (exit_time - enter_time) / 2
        annotations = [
            dict(text='', x=0.5, y=0.99, xref='paper', yref='paper', showarrow=False),
            dict(text="因子计算区间", x=enter_x, y=0.99, xref='x', yref='paper', showarrow=False, font=dict(size=14)),
            dict(text=hold_text, x=hold_x, y=0.99, xref='x', yref='paper', showarrow=False, font=dict(size=14)),
            dict(text="平仓后", x=exit_x, y=0.99, xref='x', yref='paper', showarrow=False, font=dict(size=14))
        ]
        if mark_point_list is not None:
            annotations += mark_point_list
        # , font=dict(size=15,)
        fig.update_layout(annotations=annotations)

    # 更新布局设置
    grid_color = '#fafafa'
    fig.update_layout(
        xaxis=dict(
            showgrid=True,  # 不显示垂直网格线
            gridcolor=grid_color  # 设置垂直网格线的颜色
        ),
        yaxis=dict(
            showgrid=False,  # 不显示水平网格线
            gridcolor=grid_color  # 设置水平网格线的颜色
        ),
        yaxis2=dict(
            showgrid=True,  # 显示水平网格线
            gridcolor=grid_color  # 设置水平网格线的颜色
        )
    )

    # 设置volume_bar的x轴和y轴的网格颜色
    fig.update_xaxes(showgrid=True, row=2, col=1, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, row=2, col=1, gridcolor=grid_color)

    # fig.update_layout(
    #     hovermode='x',
    #     xaxis=dict(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across+toaxis"),
    #     yaxis=dict(showspikes=False)
    # )

    if return_fig:
        # html_str = fig.to_html(full_html=False, default_height='50%')
        # with open(file_path, "w", encoding="utf-8") as file:
        #     file.write(html_str)
        return fig
    else:
        # 显示图表
        fig.show()
    return None
