# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import data

# 设置字体大小
matplotlib.rcParams.update({'font.size': 11})

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
    # for exp in data_set4:
    # data.print_array(data_set4[0], True)

    # 获取改进遗传算法的收敛用迭代次数
    generation4 = []
    for cur in data_set4:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                generation4.append(last[1])
                break
            last = data[:]

    # 获取遗传算法的收敛用迭代次数
    generation5 = []
    for cur in data_set5:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                generation5.append(last[1])
                break
            last = data[:]

    # 获取粒子群优化算法的收敛用迭代次数
    generation6 = []
    for cur in data_set6:
        temp = list(reversed(cur))
        temp_val = temp[0][0][1]
        last = None
        for data in temp:
            if data[0][1] != temp_val:
                generation6.append(last[1])
                break
            last = data[:]

    # plt.title(u'20次试验中各算法的收敛代数')

    # x坐标数组,y坐标数组,形状
    plt.plot([x for x in range(1, 21)], generation4, 'p-', label=u'Improved GA')
    plt.plot([x for x in range(1, 21)], generation5, 's-', label=u'GA')
    plt.plot([x for x in range(1, 21)], generation6, 'o-', label=u'PSO')
    # x轴范围,y轴范围
    plt.axis([1, 20, 0, 800])
    # y轴文字
    plt.ylabel(u'Iterations')
    # x轴文字
    plt.xlabel(u'Experiment number')
    # 标识
    plt.legend(loc=2, ncol=3)
    # 展示
    plt.show()
