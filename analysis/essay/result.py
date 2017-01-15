# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
from pylab import mpl
import data

# 设置字体大小
matplotlib.rcParams.update({'font.size': 11})
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    data_set1 = []
    data_set2 = []
    data_set3 = []
    data_set4 = []
    data_set5 = []
    data_set6 = []
    for index in range(1, 21):
        # path1 = '../../result/' + str(index) + '_5_immune_genetic_algorithm.pkl'
        # path2 = '../../result/' + str(index) + '_6_hybrid_genetic_algorithm.pkl'
        # path3 = '../../result/' + str(index) + '_7_improved_genetic_algorithm.pkl'
        # path4 = '../../result/' + str(index) + '_8_improved_immune_genetic_algorithm.pkl'
        # path5 = '../../result/' + str(index) + '_9_improved_particle_swarm_optimization.pkl'
        # path6 = '../../result/' + str(index) + '_10_improved_artificial_bee_colony.pkl'
        path1 = '../../result2/' + str(index) + '_5_immune_genetic_algorithm.pkl'
        path2 = '../../result2/' + str(index) + '_6_hybrid_genetic_algorithm.pkl'
        path3 = '../../result2/' + str(index) + '_7_improved_genetic_algorithm.pkl'
        path4 = '../../result2/' + str(index) + '_8_improved_immune_genetic_algorithm.pkl'
        path5 = '../../result2/' + str(index) + '_9_improved_particle_swarm_optimization.pkl'
        path6 = '../../result2/' + str(index) + '_10_improved_artificial_bee_colony.pkl'
        data_set1.append(data.read_result(path1))
        data_set2.append(data.read_result(path2))
        data_set3.append(data.read_result(path3))
        data_set4.append(data.read_result(path4))
        data_set5.append(data.read_result(path5))
        data_set6.append(data.read_result(path6))
    # for exp in data_set4:
    # data.print_array(data_set4[0], True)

    # 获取免疫遗传算法的平均结果适应度
    result1 = 0
    for cur in data_set1:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                result1 += last[0][1]
                break
            last = data[:]
    result1 /= len(data_set1)
    print('IGA: ' + str(result1))

    # 获取混合免疫算法的平均结果适应度
    result2 = 0
    for cur in data_set2:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                result2 += last[0][1]
                break
            last = data[:]
    result2 /= len(data_set2)
    print('Hybrid IGA: ' + str(result2))

    # 获取改进遗传算法的平均结果适应度
    result3 = 0
    for cur in data_set3:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                result3 += last[0][1]
                break
            last = data[:]
    result3 /= len(data_set3)
    print('Improved GA ' + str(result3))

    # 获取改进免疫遗传算法的平均结果适应度
    result4 = 0
    for cur in data_set4:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                result4 += last[0][1]
                break
            last = data[:]
    result4 /= len(data_set4)
    print('Improved IGA: ' + str(result4))

    # 获取改进粒子群算法的平均结果适应度
    result5 = 0
    for cur in data_set5:
        temp_val = temp[0][0]
        last = None
        for data in temp:
            if data[0] != temp_val:
                result5 += last[0][1]
                break
            last = data[:]
    result5 /= len(data_set5)
    print('Improved PSO: ' + str(result5))

    # 获取改进人工蜂群算法的平均结果适应度
    result6 = 0
    for cur in data_set6:
        temp = list(reversed(cur))
        temp_val = temp[0][0]
        last = None
        for data in temp:
            if data[0] != temp_val:
                result6 += last[0]
                break
            last = data[:]
    result6 /= len(data_set6)
    print('Improved ABC: ' + str(result6))
