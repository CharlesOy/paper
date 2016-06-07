# -*- coding: utf-8 -*-
import pandas as pd
import data


def qos_time(matrix_data, gene):
    """
    QoS之一,计算服务器响应时间的Web服务质量
    :param matrix_data:
    :param gene:
    :return:
    """
    score = 0
    for i in range(len(gene)):
        score += matrix_data[i][data.columns[0]][gene[i]]
    return score


test_gene = [1, 14, 18, 6, 12, 7]

if __name__ == '__main__':
    simulation_data = []
    for j in range(data.class_length):
        simulation_data.append(pd.read_csv(data.dt_path(j), index_col=0))
        print(simulation_data[j])
    print(qos_time(simulation_data, test_gene))
