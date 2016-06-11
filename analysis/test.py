# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import data

# 设置字体大小
matplotlib.rcParams.update({'font.size': 9})
plt.rc('font', serif='Songti SC')

if __name__ == '__main__':
    data_set1 = []
    data_set2 = []
    data_set3 = []
    data_set4 = []
    data_set5 = []
    data_set6 = []
    for index in range(1, 21):
        path1 = '../result/' + str(index) + '_1_random_search.pkl'
        path2 = '../result/' + str(index) + '_2_random_restart_hill_climbing.pkl'
        path3 = '../result/' + str(index) + '_3_simulated_annealing.pkl'
        path4 = '../result/' + str(index) + '_4_improved_generic_algorithm.pkl'
        path5 = '../result/' + str(index) + '_5_generic_algorithm.pkl'
        path6 = '../result/' + str(index) + '_6_particle_swarm_optimization.pkl'
        data_set1.append(data.read_result(path1))
        data_set2.append(data.read_result(path2))
        data_set3.append(data.read_result(path3))
        data_set4.append(data.read_result(path4))
        data_set5.append(data.read_result(path5))
        data_set6.append(data.read_result(path6))

    best_v = [0 for index in range(6)]
    # 获取随机搜索算法得到的最优解
    best_v[0] = data_set1[0][1]
    for l in data_set1:
        if l[1] > best_v[0]:
            best_v[0] = l[1]

    # 获取随机重复爬山算法得到的最优解
    best_v[1] = data_set2[0][1]
    for l in data_set2:
        if l[1] > best_v[1]:
            best_v[1] = l[1]

    # 获取重复模拟退火算法得到的最优解
    best_v[2] = data_set3[0][1]
    for l in data_set3:
        if l[1] > best_v[2]:
            best_v[2] = l[1]

    # 获取改进遗传算法得到的最优解
    for l in data_set4:
        best_v[3] += l[len(l) - 1][0][1]
    best_v[3] /= len(data_set4)

    # 获取遗传算法得到的最优解
    for l in data_set5:
        best_v[4] += l[len(l) - 1][0][1]
    best_v[4] /= len(data_set5)

    # 获取改进遗传算法得到的最优解
    for l in data_set6:
        best_v[5] += l[len(l) - 1][0][1]
    best_v[5] /= len(data_set6)

    # 输出各个算法得到的最优解
    print(np.round(best_v, 3))
