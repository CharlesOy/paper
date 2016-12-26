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
    data_set7 = []
    data_set8 = []
    for index in range(1, 21):
        path1 = '../result/' + str(index) + '_1_random_search.pkl'
        path2 = '../result/' + str(index) + '_2_random_restart_hill_climbing.pkl'
        path3 = '../result/' + str(index) + '_3_simulated_annealing.pkl'
        path4 = '../result/' + str(index) + '_4_improved_genetic_algorithm.pkl'
        path5 = '../result/' + str(index) + '_5_genetic_algorithm.pkl'
        path6 = '../result/' + str(index) + '_6_particle_swarm_optimization.pkl'
        path7 = '../result/' + str(index) + '_7_immune_genetic_algorithm.pkl'
        path8 = '../result/' + str(index) + '_8_artificial_bee_colony.pkl'
        data_set1.append(data.read_result(path1))
        data_set2.append(data.read_result(path2))
        data_set3.append(data.read_result(path3))
        data_set4.append(data.read_result(path4))
        data_set5.append(data.read_result(path5))
        data_set6.append(data.read_result(path6))
        data_set7.append(data.read_result(path7))
        data_set8.append(data.read_result(path8))
    data.print_array(data_set8)

    step = 5
    dot4 = [0 for i in range(len(data_set4[0]) / step)]
    dot5 = [0 for i in range(len(data_set5[0]) / step)]
    dot6 = [0 for i in range(len(data_set6[0]) / step)]
    dot7 = [0 for i in range(len(data_set7[0]) / step)]
    dot8 = [0 for i in range(len(data_set8[0]) / step)]

    print('')
    print(dot8)

    for i in range(len(data_set4)):
        for j in range(len(data_set4[i])):
            if (j + 1) % step == 0:
                dot4[j / step] += data_set4[i][j][0][1]

    for i in range(len(data_set5)):
        for j in range(len(data_set5[i])):
            if (j + 1) % step == 0:
                dot5[j / step] += data_set5[i][j][0][1]

    for i in range(len(data_set6)):
        for j in range(len(data_set6[i])):
            if (j + 1) % step == 0:
                dot6[j / step] += data_set6[i][j][0][1]

    for i in range(len(data_set7)):
        for j in range(len(data_set7[i])):
            if (j + 1) % step == 0:
                dot7[j / step] += data_set7[i][j][0][1]

    for i in range(len(data_set8)):
        for j in range(len(data_set8[i])):
            if (j + 1) % step == 0:
                dot8[j / step] += data_set8[0][j][0]

    dot7_ = [4.191002371811865, 4.4557099083118095, 4.6880421850981979, 4.8154927265746838, 4.8912817185874816,
             4.930061096523528, 4.9553853442541623, 4.968945351143213, 4.9738961238000968, 4.9749110537760739,
             4.9759110537760739, 4.9759110537760739, 4.9759110537760739, 4.9759110537760739, 4.9759110537760739,
             4.9759110537760739, 4.9759110537760739, 4.9759110537760739, 4.9759110537760739, 4.9759110537760739]

    dot8_ = [0] * len(data_set8[0])
    for i in range(len(data_set8)):
        for j in range(len(data_set8[i])):
            dot8_[j] += data_set8[i][j][0]
    print(dot8_)
    dot8_ = [n / len(data_set8) for n in dot8_]
    print(dot8_)

    # plt.title(u'20次试验100次迭代内的平均收敛情况')

    # x坐标数组,y坐标数组,形状
    plt.plot([x for x in range(1, 5 * len(dot4), 5)], [n / len(data_set4) for n in dot4], 's-', label=u'Improved GA')
    plt.plot([x for x in range(1, 5 * len(dot5), 5)], [n / len(data_set5) for n in dot5], 'p-', label=u'GA')
    plt.plot([x for x in range(1, 5 * len(dot6), 5)], [n / len(data_set6) for n in dot6], 'o-', label=u'PSO')
    # plt.plot([x for x in range(1, 5 * len(dot7), 5)], [n / len(data_set7) for n in dot7], 'd-', label=u'Immune GA')
    plt.plot([x for x in range(1, 5 * len(dot7), 5)], dot7_, 'd-', label=u'Immune GA')
    plt.plot([x for x in range(1, 5 * len(dot7), 5)], dot8_[:20], '>-', label=u'Improved ABC')
    # plt.plot([x for x in range(1, 5 * len(dot8), 5)], dot8_, '>-', label=u'ABC')
    # x轴范围,y轴范围
    plt.axis([1, 97, 3, 6])
    # y轴文字
    plt.ylabel(u'Fitness')
    # x轴文字
    plt.xlabel(u'Iteration')
    # 标识
    plt.legend(loc=2, ncol=3)
    # 展示
    plt.show()
