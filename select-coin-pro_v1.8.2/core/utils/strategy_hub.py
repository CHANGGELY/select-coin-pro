"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import hashlib
import importlib

import pandas as pd


class DummyStrategy:
    """
    ！！！！抽象策略对象，用于代码提示！！！！
    """

    def __init__(self):
        self.name = ''
        # 持仓周期。目前回测支持日线级别、小时级别。例：1H，6H，3D，7D......
        # 当持仓周期为D时，选币指标也是按照每天一根K线进行计算。
        # 当持仓周期为H时，选币指标也是按照每小时一根K线进行计算。
        self.hold_period = '24h'.replace('H', 'h').replace('d', 'D')

        # 是否使用现货
        self.if_use_spot = True  # True：使用现货。False：不使用现货，只使用合约。

        # 选币市场范围 & 交易配置
        #   配置解释： 选币范围 + '_' + 优先交易币种类型
        #
        #   spot_spot: 在 '现货' 市场中进行选币。如果现货币种含有'合约'，优先交易 '现货'。
        #   swap_swap: 在 '合约' 市场中进行选币。如果现货币种含有'现货'，优先交易 '合约'。
        #   spot_swap: 在 '现货' 市场中进行选币。如果现货币种含有'合约',优先交易'合约'。
        #   mix_spot:  在 '现货与合约' 的市场中进行选币。如果两边市场都存在的币种，会保留'现货'，并优先下单'现货'。
        #   mix_swap:  在 '现货与合约' 的市场中进行选币。如果两边市场都存在的币种，会保留'合约'，并优先下单'合约'。
        self.markets = 'spot_spot'

        # 配置offset
        self.offset = 0

        # 多头选币数量。1 表示做多一个币; 0.1 表示做多10%的币
        self.long_select_coin_num = 0.1
        # 空头选币数量。1 表示做空一个币; 0.1 表示做空10%的币
        # short_select_coin_num = 0.1
        self.short_select_coin_num = 'long_nums'  # long_nums意为着空头数量和多头数量保持一致。最多为所有合约的数量。

        # 多头的选币因子列名。
        self.long_factor = '因子'  # 因子：表示使用复合因子，默认是 factor_list 里面的因子组合。需要修改 calc_factor 函数配合使用
        # 空头的选币因子列名。多头和空头可以使用不同的选币因子
        self.short_factor = '因子'

        # 选币因子信息列表，用于`2_选币_单offset.py`，`3_计算多offset资金曲线.py`共用计算资金曲线
        self.factor_list = []

        # 确认过滤因子及其参数，用于`2_选币_单offset.py`进行过滤
        self.filter_list = []

        self.is_abstract = True

        self.md5_hash = ''

    def after_merge_index(self, df, symbol, factor_dict, data_dict) -> (pd.DataFrame, dict, dict):
        raise NotImplementedError

    def after_resample(self, df, symbol) -> pd.DataFrame:
        raise NotImplementedError

    def calc_factor(self, df, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def before_filter(self, df, **kwargs) -> (pd.DataFrame, pd.DataFrame):
        raise NotImplementedError


class StrategyHub:
    _strategy_cache = {}

    # noinspection PyTypeChecker,PyUnresolvedReferences
    @staticmethod
    def get_by_name(strategy_name) -> DummyStrategy:
        if strategy_name in StrategyHub._strategy_cache:
            return StrategyHub._strategy_cache[strategy_name]

        try:
            # 构造模块名
            module_name = f"strategy.{strategy_name}"

            # 动态导入模块
            strategy_module = importlib.import_module(module_name)

            # 创建一个包含模块变量和函数的字典
            strategy_content = {
                name: getattr(strategy_module, name) for name in dir(strategy_module)
                if not name.startswith("__")
            }

            # 创建一个包含这些变量和函数的对象
            strategy_instance = type(strategy_name, (), strategy_content)
            strategy_instance.name = strategy_name
            strategy_instance.is_use_spot = getattr(strategy_instance, 'if_use_spot', False)
            strategy_instance.markets = 'spot_spot' if strategy_instance.is_use_spot else 'swap_swap'

            strategy_instance.is_abstract = False

            # ** 回测特有 ** 缓存当前文件的md5值，优化重复计算的过程
            strategy_instance.md5_hash = hashlib.md5(str(sorted(strategy_content)).encode('utf-8')).hexdigest()

            # 缓存策略对象
            StrategyHub._strategy_cache[strategy_name] = strategy_instance

            return strategy_instance
        except ModuleNotFoundError:
            dummy_strategy = DummyStrategy()
            dummy_strategy.name = strategy_name
            dummy_strategy.strategy = strategy_name
            return dummy_strategy
            # raise ValueError(f"Strategy {strategy_name} not found.")
        except AttributeError:
            raise ValueError(f"Error accessing strategy content in module {strategy_name}.")


# 使用示例
if __name__ == "__main__":
    strategy = StrategyHub.get_by_name("Strategy_Spot_100")
    print(strategy.long_factor)  # 访问变量
    print(strategy.factor_list)  # 访问变量
    print(strategy.after_resample('xxx', 's'))
