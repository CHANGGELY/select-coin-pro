import pandas as pd
import numpy as np


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['Zscore'] = (df['close'] - df['close'].rolling(n, min_periods=1).mean()) / df['close'].rolling(n, min_periods=1).std()
    df['ZscoreAbsMean'] = df['Zscore'].abs().rolling(n, min_periods=1).mean()

    df[factor_name] = df['ZscoreAbsMean'].rolling(n, min_periods=1).rank(ascending=True, pct=True)

    return df
