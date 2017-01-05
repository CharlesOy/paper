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
        path1 = '../../result/' + str(index) + '_5_immune_genetic_algorithm.pkl'
        path2 = '../../result/' + str(index) + '_6_hybrid_genetic_algorithm.pkl'
        path3 = '../../result/' + str(index) + '_7_improved_genetic_algorithm.pkl'
        path4 = '../../result/' + str(index) + '_8_improved_immune_genetic_algorithm.pkl'
        path5 = '../../result/' + str(index) + '_9_improved_particle_swarm_optimization.pkl'
        path6 = '../../result/' + str(index) + '_10_improved_artificial_bee_colony.pkl'
        data_set1.append(data.read_result(path1))
        data_set2.append(data.read_result(path2))
        data_set3.append(data.read_result(path3))
        data_set4.append(data.read_result(path4))
        data_set5.append(data.read_result(path5))
        data_set6.append(data.read_result(path6))
    # for exp in data_set4:
    # data.print_array(data_set4[0], True)

    # 获取免疫遗传算法的收敛用时
    time1 = []
    for cur in data_set1:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                time1.append(last[2])
                break
            last = data[:]

    # 获取混合免疫算法的收敛用时
    time2 = []
    for cur in data_set2:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                time2.append(last[2])
                break
            last = data[:]

    # 获取改进遗传算法的收敛用时
    time3 = []
    for cur in data_set3:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                time3.append(last[2])
                break
            last = data[:]

    # 获取改进免疫遗传算法的收敛用时
    time4 = []
    for cur in data_set4:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                time4.append(last[2])
                break
            last = data[:]

    # 获取改进粒子群算法的收敛用时
    time5 = []
    for cur in data_set5:
        temp = list(reversed(cur))
        temp_val = temp[0][0]
        last = None
        for data in temp:
            if data[0] != temp_val:
                time5.append(last[2])
                break
            last = data[:]

    # 获取改进人工蜂群算法的收敛用时
    time6 = []
    for cur in data_set6:
        temp = list(reversed(cur))
        temp_val = temp[0][0]
        last = None
        for data in temp:
            if data[0] != temp_val:
                time6.append(last[2])
                break
            last = data[:]

    # plt.title(u'20次试验中各算法的平均(收敛)用时')

    # x坐标数组,y坐标数组,形状
    # plt.plot([x for x in range(1, 21)], [data_set1[y][2] for y in range(0, 20)], '<-', label=u'Random Search')
    # plt.plot([x for x in range(1, 21)], [data_set2[y][2] for y in range(0, 20)], '>-', label=u'Hill Climbing')
    plt.plot([x for x in range(1, 21)], time1, 's-', label=u'IGA')
    plt.plot([x for x in range(1, 21)], time2, 'p-', label=u'Hybrid IGA')
    plt.plot([x for x in range(1, 21)], time3, 'o-', label=u'Improved GA')
    plt.plot([x for x in range(1, 21)], time4, 'd-', label=u'Improved IGA')
    plt.plot([x for x in range(1, 21)], time5, '>-', label=u'Improved PSO')
    plt.plot([x for x in range(1, 21)], time6, 'v-', label=u'Improved ABC')
    # x轴范围,y轴范围
    plt.axis([1, 20, 0, 1000])
    # y轴文字
    plt.ylabel(u'Time(s)')
    # x轴文字
    plt.xlabel(u'Experiment number')
    # 标识
    plt.legend(loc=2, ncol=3)
    # 展示
    plt.show()
