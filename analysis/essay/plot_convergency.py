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

    step = 5
    dot1 = [0 for i in range(int(len(data_set1[0]) / step))]
    dot2 = [0 for i in range(int(len(data_set2[0]) / step))]
    dot3 = [0 for i in range(int(len(data_set3[0]) / step))]
    dot4 = [0 for i in range(int(len(data_set4[0]) / step))]
    dot5 = [0 for i in range(int(len(data_set5[0]) / step))]
    dot6 = [0 for i in range(int(len(data_set5[0]) / step))]

    for i in range(len(data_set1)):
        for j in range(len(data_set1[i])):
            if (j + 1) % step == 0:
                dot1[int(j / step)] += data_set1[i][j][0][1]

    for i in range(len(data_set2)):
        for j in range(len(data_set2[i])):
            if (j + 1) % step == 0:
                dot2[int(j / step)] += data_set2[i][j][0][1]

    for i in range(len(data_set3)):
        for j in range(len(data_set3[i])):
            if (j + 1) % step == 0:
                dot3[int(j / step)] += data_set3[i][j][0][1]

    for i in range(len(data_set4)):
        for j in range(len(data_set4[i])):
            if (j + 1) % step == 0:
                dot4[int(j / step)] += data_set4[i][j][0][1]

    for i in range(len(data_set5)):
        for j in range(len(data_set5[i])):
            if (j + 1) % step == 0:
                dot5[int(j / step)] += data_set5[i][j][0][1]

    for i in range(len(data_set6)):
        for j in range(len(data_set6[i])):
            dot6[j] += data_set6[i][j][0]

    # plt.title(u'20次试验100次迭代内的平均收敛情况')

    # x坐标数组,y坐标数组,形状
    plt.plot([x for x in range(1, 5 * len(dot1), 5)], [n / len(data_set1) for n in dot1], 's-', label=u'IGA')
    plt.plot([x for x in range(1, 5 * len(dot2), 5)], [n / len(data_set2) for n in dot2], 'p-', label=u'Hybrid IGA')
    plt.plot([x for x in range(1, 5 * len(dot3), 5)], [n / len(data_set3) for n in dot3], 'o-', label=u'Improved GA')
    plt.plot([x for x in range(1, 5 * len(dot4), 5)], [n / len(data_set4) for n in dot4], 's-', label=u'Improved IGA')
    plt.plot([x for x in range(1, 5 * len(dot5), 5)], [n / len(data_set5) for n in dot5], 'p-', label=u'Improved PSO')
    plt.plot([x for x in range(1, 5 * len(dot6), 5)], [n / len(data_set6) for n in dot6], 'o-', label=u'Improved ABC')
    # plt.plot([x for x in range(1, 5 * len(dot7), 5)], [n / len(data_set7) for n in dot7], 'd-', label=u'Immune GA')
    # plt.plot([x for x in range(1, 5 * len(dot7), 5)], dot7_, 'd-', label=u'Immune GA')
    # plt.plot([x for x in range(1, 5 * len(dot7), 5)], dot8_[:20], '>-', label=u'Improved ABC')
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
