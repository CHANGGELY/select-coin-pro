# -*- coding: utf-8 -*-
"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import copy
import itertools
import json
import operator
import os
import re
import warnings
from functools import reduce
from pathlib import Path
import pandas as pd

import tools.utils.pfunctions as pf
from core.utils.path_kit import get_folder_path, get_file_path
from tools.utils.unified_tool import UnifiedToolParam

warnings.filterwarnings("ignore")


# ====== 公共函数 ======
def dict_itertools(dict_):
    keys = list(dict_.keys())
    values = list(dict_.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def filter_dataframe(df, filter_dict):
    conditions = [df[col].isin(values) for col, values in filter_dict.items()]
    return df[reduce(operator.and_, conditions)] if conditions else df.copy()


def prepare_data():
    """生成参数组合并过滤"""
    params_df = pd.DataFrame(dict_itertools(batch))
    params_df["参数组合"] = [f"参数组合_{i + 1}" for i in range(len(params_df))]
    df = filter_dataframe(params_df, limit_dict)
    # pivot的时候不支持list，所以此处把list改为str
    for k, v in batch.items():
        if any(isinstance(_, list) for _ in v):
            df[k] = df[k].astype("str")
    return df


def load_and_process_data(df_left, result_dir: Path):
    """加载并处理策略评价数据"""
    if evaluation_indicator not in [
        "累积净值",
        "年化收益",
        "最大回撤",
        "年化收益/回撤比",
        "盈利周期数",
        "亏损周期数",
        "胜率",
        "每周期平均收益",
        "盈亏收益比",
        "单周期最大盈利",
        "单周期大亏损",
        "最大连续盈利周期数",
        "最大连续亏损周期数",
        "收益率标准差",
    ]:
        raise ValueError("评价指标有误，按要求输入")

    if evaluation_indicator == "年化收益":
        time_list = []
        for folder in df_left["参数组合"]:
            # 读取策略评价数据
            stats_path = result_dir / folder / "策略评价.csv"
            stats_temp = pd.read_csv(stats_path, encoding="utf-8")
            stats_temp.columns = ["evaluation_indicator", "value"]
            if stats_temp.empty:
                raise ValueError(f"{folder} 文件夹内策略评价数据为空，请检查数据")
            stats_temp = stats_temp.set_index("evaluation_indicator")
            df_left.loc[df_left["参数组合"] == folder, "all"] = stats_temp.loc[evaluation_indicator, "value"]

            # 读取年度数据
            years_path = result_dir / folder / "年度账户收益.csv"
            years_return = pd.read_csv(years_path, encoding="utf-8")
            if years_return.empty:
                raise ValueError(f"{folder} 文件夹内年度账户收益数据为空，请检查数据")
            time_list = list(years_return["candle_begin_time"].sort_values(ascending=False))
            for time in time_list:
                df_left.loc[df_left["参数组合"] == folder, time] = years_return.loc[
                    years_return["candle_begin_time"] == time, "涨跌幅"
                ].iloc[0]

        # 格式转换
        df_left[["all"] + time_list] = df_left[["all"] + time_list].map(
            lambda x: float(x.replace("%", "")) / 100 if "%" in str(x) else float(x)
        )
        return time_list
    else:
        for folder in df_left["参数组合"]:
            stats_path = result_dir / folder / "策略评价.csv"
            stats_temp = pd.read_csv(stats_path, encoding="utf-8")
            if stats_temp.empty:
                raise ValueError(f"{folder} 文件夹内策略评价数据为空，请检查数据")
            stats_temp.columns = ["evaluation_indicator", "value"]
            stats_temp = stats_temp.set_index("evaluation_indicator")
            df_left.loc[df_left["参数组合"] == folder, evaluation_indicator] = stats_temp.loc[
                evaluation_indicator, "value"
            ]

        df_left[evaluation_indicator] = df_left[evaluation_indicator].apply(
            lambda x: float(x.replace("%", "")) / 100 if "%" in str(x) else float(x)
        )
        return None


def generate_plots(df_left, params, output_dir: Path, analysis_type, time_list):
    """根据分析类型生成图表"""
    fig_list = []
    html_name = f"年化收益_回撤比.html" if evaluation_indicator == "年化收益/回撤比" else f"{evaluation_indicator}.html"

    if "hold_period" in df_left.columns:
        df_left["periods"] = df_left["hold_period"].apply(lambda x: int(x[:-1]))
        df_left = df_left.sort_values(by=["periods"])

    if analysis_type == "double":
        x_, y_ = params

        if evaluation_indicator == "年化收益":
            for time in ["all"] + time_list:
                temp = pd.pivot_table(df_left, index=y_, columns=x_, values=time)
                fig = pf.draw_params_heatmap_plotly(temp, title=time)
                fig_list.append(fig)
        else:
            temp = pd.pivot_table(df_left, index=y_, columns=x_, values=evaluation_indicator)
            fig = pf.draw_params_heatmap_plotly(temp, title=evaluation_indicator)
            fig_list.append(fig)
        html_name = f"{x_}_{y_}_{html_name}"

    else:
        param = params
        if evaluation_indicator == "年化收益":
            sub_df = df_left[[param] + ["all"] + time_list].copy()
            sub_df[param] = sub_df[param].map(lambda x: f"{param}_{x}")
            sub_df = sub_df.set_index(param)
            fig = pf.draw_params_bar_plotly(sub_df, evaluation_indicator)
        else:
            x_axis = df_left[param].map(lambda x: f"{param}_{x}")
            fig = pf.draw_bar_plotly(
                x_axis, df_left[evaluation_indicator], title=evaluation_indicator, pic_size=[1800, 600]
            )
        fig_list.append(fig)
        html_name = f"{param}_{html_name}"

    if fig_list:
        title = "参数热力图" if analysis_type == "double" else "参数平原图"
        pf.merge_html_flexible(fig_list, output_dir / html_name, title=title)

    return output_dir / html_name


# ====== 主逻辑 ======
def analyze_params(analysis_type):
    """参数分析主函数"""
    df_left = prepare_data()

    # 配置输出路径
    if analysis_type == "double":
        output_dir = out_folder_path / "参数热力图" / trav_name
        params = [param_x, param_y]
    else:
        output_dir = out_folder_path / "参数平原图" / trav_name
        params = param_x
    os.makedirs(output_dir, exist_ok=True)

    # 处理数据
    time_list = load_and_process_data(df_left, result_folder_path)

    # 生成图表
    html_path = generate_plots(df_left, params, output_dir, analysis_type, time_list)

    return dict(
        name="参数热力图" if analysis_type == "double" else "参数平原图",
        html=str(html_path).split(f'{os.path.sep}分析结果')[1],
        full_path=str(html_path)
    )


def __load_and_clean_config():
    """加载并清理配置文件

    Returns:
        tuple: (trav_name, strategy_list, batch)
    """
    with open(get_file_path("config.json"), "r", encoding="utf-8") as f:
        data_json = json.load(f)

    search_name = data_json.get("search_name", "遍历")
    stg = data_json.get("strategy_info", {})

    # 清理batch数据：移除name、search_name、strategy_info这三个key，以及value不存在的数据
    keys_to_remove = ["name", "search_name", "strategy_info"]
    for key in keys_to_remove:
        data_json.pop(key, None)  # 安全移除，如果key不存在不会报错

    # 移除value不存在的数据（None、空字符串、空list等）
    data_json = {
        k: v
        for k, v in data_json.items()
        if v is not None and v != "" and v != [] and v != {}
    }

    return search_name, stg, data_json


def __parse_path_expression(path_expr):
    """解析路径表达式，如 'factor_list[0][2][0]'

    Args:
        path_expr: 路径表达式字符串

    Returns:
        tuple: (base_key, indices)
        - base_key: 基础键名，如 'factor_list'
        - indices: 索引列表，如 [0, 2, 0]
    """
    # 使用正则表达式匹配基础键名和所有索引
    match = re.match(r"^([^[]+)((?:\[\d+\])+)$", path_expr)
    if not match:
        return path_expr, []

    base_key = match.group(1)
    indices_str = match.group(2)

    # 提取所有数字索引
    indices = [int(idx) for idx in re.findall(r"\[(\d+)\]", indices_str)]

    return base_key, indices


def __resolve_factor_name(path_expr, strategy_info):
    """解析路径表达式并获取因子名称

    Args:
        path_expr: 路径表达式字符串，如 'factor_list[0][2][1]'
        strategy_info: 策略信息字典

    Returns:
        str or None: 解析后的因子名称（带后缀），如果解析失败返回None
    """
    base_key, indices = __parse_path_expression(path_expr)

    if base_key and len(indices) >= 2:
        # 获取 factor_list 或 filter_list
        factor_list = strategy_info.get(base_key, [])

        if len(factor_list) > indices[0]:
            # 获取因子配置 factor_list[0]
            factor_config = factor_list[indices[0]]

            # 获取因子名称 factor_list[0][0]
            factor_name = factor_config[0]

            # 解决一个因子的参数是列表的问题
            factor_name = f"{factor_name}_{indices[0]}_{indices[-1]}"

            return factor_name
        else:
            print(f"Warning: 未检测到因子配置 {path_expr}")
            return None
    else:
        # 不是路径表达式
        return None


def convert_lists_to_tuples(data, target_fields=None):
    """将指定字段中的列表里的列表转换为元组

    Args:
        data: 字典数据
        target_fields: 需要处理的字段集合，默认为None时处理所有字段

    Returns:
        处理后的数据
    """
    if not isinstance(data, dict):
        return data

    # 默认的元组字段
    if target_fields is None:
        target_fields = {
            "factor_list",
            "long_factor_list",
            "short_factor_list",
            "filter_list",
            "long_filter_list",
            "short_filter_list",
            "filter_list_post",
            "long_filter_list_post",
            "short_filter_list_post",
        }

    # 深拷贝以避免修改原数据
    result = copy.deepcopy(data)

    for field in target_fields:
        if field in result and isinstance(result[field], list):
            result[field] = [
                tuple(item) if isinstance(item, list) else item
                for item in result[field]
            ]

    return result


def convert_range_params(data):
    """转换range格式的参数为列表

    Args:
        data: 配置数据（通常是字典）

    Returns:
        转换后的数据
    """
    if isinstance(data, dict):
        # 检查是否是range格式 {"start": x, "end": y, "step": z}
        if all(key in data for key in ["start", "end", "step"]):
            start = data["start"]
            end = data["end"]
            step = data["step"]
            return list(range(start, end, step))
        else:
            # 递归处理字典中的每个值
            return {k: convert_range_params(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 递归处理列表中的每个元素
        return [convert_range_params(item) for item in data]
    else:
        # 其他类型直接返回
        return data


if __name__ == "__main__":
    # ====== 配置信息 ======
    trav_name = "低价币中性策略"  # 用于读取 data/遍历结果/ 中的遍历回测结果
    # 回测路径和参数分析输出路径
    result_folder_path = get_folder_path("data", "遍历结果", trav_name, auto_create=False, as_path_type=True)
    out_folder_path = get_folder_path("data", "分析结果", "参数分析", as_path_type=True)

    # 参数设置
    batch = {"hold_period": ["24H"], "LowPrice": [48, 96, 144], "QuoteVolumeMean": [48, 96]}
    # 若绘制单参数平原图，param_x 填写变量，param_y=''
    # 若绘制双参数热力图，则 param_x和param_y 填写变量, param_为热力图x轴变量，param_y为热力图y轴变量，可按需更改
    param_x = "LowPrice"
    param_y = "QuoteVolumeMean"

    # 这里需要固定非观测参数，然后画参数图，例如该案例固定hold_period== 12H，来看LowPrice和QuoteVolumeMean的参数热力图
    # 注意点：多参数画图，必须固定其他参数。单参数平原需固定该参数以外的其他参数，双参数热力图需固定除两参数以外的参数
    limit_dict = {
        # 'rebalance_time': ["0935-0945"],
        "hold_period": ["24H"],
        # '换手率': [20],
        # '换手率': [20],
    }

    # 分析指标，支持以下：
    # 累积净值、年化收益、最大回撤、年化收益/回撤比、盈利周期数、亏损周期数、胜率、每周期平均收益
    # 盈亏收益比、单周期最大盈利、单周期大亏损、最大连续盈利周期数、最大连续亏损周期数、收益率标准差
    evaluation_indicator = "年化收益"

    # ====== 主逻辑 ======
    INPUT = [
        {
            "name": "param_search_info",
            "reflect": "param_search_info",
        }
    ]
    param_search_info = {}
    unified_tool_param = UnifiedToolParam(name=Path(__file__).stem)
    ui_input = unified_tool_param.get_input_json()
    if ui_input:
        for input_param in INPUT:
            attr_name = input_param["name"]
            ui_name = input_param["reflect"]
            globals()[attr_name] = ui_input[ui_name]

    # 遍历的策略信息
    if param_search_info:
        trav_name, strategy_info, batch = __load_and_clean_config()
        strategy_info = convert_lists_to_tuples(strategy_info)
        # 重置一下文件路径
        result_folder_path = get_folder_path(
            "data", "遍历结果", trav_name, auto_create=False, as_path_type=True
        )
        for _key, _value in param_search_info.items():

            if _key in ["param_x", "param_y"]:
                # 尝试解析路径表达式并获取因子名称
                factor_name = __resolve_factor_name(_value, strategy_info)

                if factor_name:
                    # 成功解析为因子名称
                    globals()[_key] = factor_name
                else:
                    # 不是路径表达式或解析失败，直接赋值
                    globals()[_key] = _value
            elif _key == "limit_dict":
                # 处理 limit_dict：解析路径表达式，用因子名称替换原来的key
                processed_limit_dict = {}
                for limit_key, limit_value in _value.items():
                    # 尝试解析路径表达式并获取因子名称
                    factor_name = __resolve_factor_name(limit_key, strategy_info)

                    if factor_name:
                        # 用因子名称作为新的key
                        processed_limit_dict[factor_name] = limit_value
                    else:
                        # 如果不是路径表达式或解析失败，保持原key
                        processed_limit_dict[limit_key] = limit_value

                globals()[_key] = processed_limit_dict
            else:
                globals()[_key] = _value

        # 处理batch中的路径表达式key，转换为因子名称
        processed_batch = {}
        for batch_key, batch_value in batch.items():
            # 尝试解析路径表达式并获取因子名称
            factor_name = __resolve_factor_name(batch_key, strategy_info)

            if factor_name:
                # 用因子名称作为新的key
                processed_batch[factor_name] = convert_range_params(batch_value)
            else:
                # 如果不是路径表达式或解析失败，保持原key
                processed_batch[batch_key] = convert_range_params(batch_value)

        # 更新全局的batch变量
        batch = processed_batch

    # 进行参数分析
    analysis_type = "single" if len(param_y.strip()) == 0 else "double"
    ui_output = analyze_params(analysis_type)

    unified_tool_param.save_output_json([ui_output])
