# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd


def base_data():
    """根据基本数据特征生成实验数据"""
    return pd.read_csv('data.txt', header=None)


def generate_df(_length=100):
    """生成服从正态分布的数据"""
    columns = ['ResponseTime', 'Throughput', 'Reliability', 'BestPractices', 'Documentation']

    df = base_data()

    data = []
    for i in range(len(columns)):
        sigma = math.sqrt(df[i].describe()['std'])
        mu = df[i].describe()['mean']
        data.append(pd.DataFrame(round(sigma ** 2) * np.random.randn(_length, 1) + mu))

    df_result = pd.concat(data, axis=1)
    df_result.columns = columns
    return df_result


print(generate_df())
