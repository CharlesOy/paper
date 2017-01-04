# -*- coding: utf-8 -*-
import pickle
import math
import numpy as np
import pandas as pd

# 数据维度
class_length = 30
# 每一维数据的长度
length = 10
# 列名
columns = ['ResponseTime', 'Cost', 'Availability', 'Reliability', 'Reputation']


def base_data():
    """
    获取真实数据作为基本数据
    :return:
    """
    return pd.read_csv('data/base.data', header=None)


def normalize(df):
    """
    归一化数据
    :param df:
    :return:
    """
    for i in range(len(columns)):
        if i == 0 or i == 1:
            # 响应时间和花费是越少越好,这里将其处理成原始数据越小,归一化之后的数据越大,便于后续处理
            df[columns[i]] = np.round(
                (df[columns[i]].max() - df[columns[i]]) / (df[columns[i]].max() - df[columns[i]].min()), 2)
        elif i == 4:
            # 名誉度正常归一化处理
            df[columns[i]] = np.round(
                (df[columns[i]] - df[columns[i]].min()) / (df[columns[i]].max() - df[columns[i]].min()), 2)
        else:
            # 可用性,可靠性归一化等比例放缩,最小值为0.5,最大值为1
            df[columns[i]] = np.round(((df[columns[i]] - df[columns[i]].min()) / (
                df[columns[i]].max() - df[columns[i]].min())) / 2 + 0.5, 2)


def generate_df(_length=100):
    """
    生成服从正态分布的数据,
    并写入文件
    :param _length:
    :return:
    """

    df = base_data()

    data = []
    for i in range(len(columns)):
        sigma = math.sqrt(df[i].describe()['std'])
        mu = df[i].describe()['mean']
        raw_data = pd.DataFrame(np.round(round(sigma ** 2) * np.random.randn(_length, 1) + mu, decimals=0))
        data.append(raw_data)

    df_result = pd.concat(data, axis=1)
    df_result.columns = columns

    # 归一化数据
    normalize(df_result)

    return df_result


def dt_path(i):
    """
    获取仿真数据文件路径
    :param i:
    :return:
    """
    base_path = 'data/simulation_class'
    return base_path + str(i) + '.data'


def generate_population(matrix_data, population_size):
    """
    生成初始种群,
    将生成初始种群的函数独立出来,
    以便存储初始种群,
    用于做不同算法之间的对比
    :param matrix_data:
    :param population_size:
    :return:
    """
    population = []
    for i in range(population_size):
        vec = [np.random.randint(0, matrix_data[k].shape[0])
               for k in range(len(matrix_data))]
        population.append(vec)
    df = pd.DataFrame(population)
    df.to_csv('data/population.data')
    return population


def get_population():
    """
    读取种群数据
    :return:
    """
    return [list(t) for t in pd.read_csv('data/population.data', index_col=0).to_records(index=False)]


def get_simulation_data():
    """
    获取初始种群
    :return:
    """
    result = []
    for i in range(class_length):
        result.append(pd.read_csv(dt_path(i), index_col=0))
    return result


def write_result(_path, data):
    """
    写结果文件到指定路径
    :param _path: 写文件路径
    :param data: 数据
    :return:
    """
    file_output = open(_path, 'wb')
    pickle.dump(data, file_output)
    file_output.close()
    return


def read_result(_path):
    """
    从指定路径读结果文件
    :param _path:
    :return:
    """
    file_input = open(_path, 'rb')
    data = pickle.load(file_input)
    file_input.close()
    return data


def print_array(arr, reverse=False):
    """
    打印数组
    :param arr:
    :param reverse:
    :return:
    """
    if reverse:
        for l in reversed(arr):
            print(l)
        return
    for l in arr:
        print(l)


if __name__ == '__main__':
    for j in range(class_length):
        path = dt_path(j)
        simulation_data = generate_df(_length=length)
        simulation_data.to_csv(path)
    generate_population(get_simulation_data(), 200)
