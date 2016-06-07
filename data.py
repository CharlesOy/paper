# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd


def base_data():
    """
    获取真实数据作为基本数据
    """
    return pd.read_csv('data/base.data', header=None)


def generate_df(_length=100):
    """
    生成服从正态分布的数据,
    并写入文件
    """
    # 列名
    columns = ['ResponseTime', 'Throughput', 'Reliability', 'BestPractices', 'Documentation']

    df = base_data()

    data = []
    for i in range(len(columns)):
        sigma = math.sqrt(df[i].describe()['std'])
        mu = df[i].describe()['mean']
        data.append(pd.DataFrame(round(sigma ** 2) * np.random.randn(_length, 1) + mu))

    df_result = pd.concat(data, axis=1)
    df_result.columns = columns

    df_result.to_csv('data/simulation.data')

    return df_result


print(generate_df(_length=1000))
