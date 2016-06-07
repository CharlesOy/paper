# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import operator
import data


def qos_time(matrix_data, gene):
    """
    QoS之一,计算Web服务质量的服务器响应时间因素
    :param matrix_data:
    :param gene:
    :return:
    """
    score = 0
    for i in range(len(gene)):
        score += matrix_data[i][data.columns[0]][gene[i]]
    return score


def qos_cost(matrix_data, gene):
    """
    QoS之一,计算Web服务质量的价格成本因素
    :param matrix_data:
    :param gene:
    :return:
    """
    score = 0
    for i in range(len(gene)):
        score += matrix_data[i][data.columns[1]][gene[i]]
    return score


def qos_availability(matrix_data, gene):
    """
    QoS之一,计算Web服务质量的可用性因素
    :param matrix_data:
    :param gene:
    :return:
    """
    score = 1
    for i in range(len(gene)):
        score *= matrix_data[i][data.columns[2]][gene[i]]
    return score


def qos_reliability(matrix_data, gene):
    """
    QoS之一,计算Web服务质量的可靠性因素
    :param matrix_data:
    :param gene:
    :return:
    """
    score = 1
    for i in range(len(gene)):
        score *= matrix_data[i][data.columns[3]][gene[i]]
    return score


def qos_reputation(matrix_data, gene):
    """
    QoS之一,计算Web服务质量的名誉度因素
    :param matrix_data:
    :param gene:
    :return:
    """
    score = 1
    for i in range(len(gene)):
        score += matrix_data[i][data.columns[3]][gene[i]]
    return score / len(gene)


def qos_total(matrix_data, gene):
    """
    线性加权计算QoS总质量,暂定权值均为0.2,满足权值和为1
    :param matrix_data:
    :param gene:
    :return:
    """
    result = 0.2 * qos_time(matrix_data, gene)
    result += 0.2 * qos_cost(matrix_data, gene)
    result += 0.2 * qos_availability(matrix_data, gene)
    result += 0.2 * qos_reliability(matrix_data, gene)
    result += 0.2 * qos_reputation(matrix_data, gene)
    return result


def best_qos(matrix_data):
    """
    获取最佳服务质量及其解,
    本函数采用穷举法,及其消耗时间
    :param matrix_data:
    :return:
    """

    def self_plus(index=-2):
        """
        自加1
        :param index:
        :return:
        """
        if index == -1:
            # 迭代结束
            return False
        if index == -2:
            index = len(vec) - 1
        while index >= 0:
            if vec[index] + 1 >= matrix_data[index].shape[0]:
                vec[index] = 0
                self_plus(index - 1)
                break
            else:
                vec[index] += 1
                break
        return True

    # vec = [19, 19, 19, 19, 19, 19]
    # self_plus()
    # if self_plus():
    #     print('test')
    # print(vec)

    vec = []
    vec_pre = []
    vec_best = []
    adaption_best = 0
    for i in range(len(matrix_data)):
        vec.append(0)
        vec_best.append(0)
        vec_pre.append(-1)

    while self_plus():
        adaption_cur = qos_total(matrix_data, vec)
        if adaption_cur > adaption_best:
            adaption_best = adaption_cur
            vec_best = vec[:]
        if vec_pre[2] != vec[2]:
            print(vec_best, adaption_best)
            print(vec)
        vec_pre = vec[:]
    return vec_best, adaption_best


def genetic_optimize(matrix_data, adaptive_function, mutate_prob=0.2, population_size=100, step=1, elite_rate=0.2,
                     max_iter=100):
    """
    经典遗传算法,
    采用选择算子采用排序法,
    变异算子,
    交叉算子采用随机交叉
    :param matrix_data:
    :param adaptive_function:
    :param mutate_prob:
    :param population_size:
    :param step:
    :param elite_rate:
    :param max_iter:
    :return:
    """

    def mutate(vec):
        """
        变异算子
        :param vec:
        :return:
        """
        i = np.random.randint(0, len(matrix_data))
        if np.random.random() < 0.5:
            # 染色体变异位置超过下限
            if vec[i] - step < 0:
                return vec[0:i] + [0] + vec[i + 1:]
            # 向下变异,移动step个单位
            return vec[0:i] + [vec[i] - step] + vec[i + 1:]
        else:
            # 染色体变异位置超过上限
            if vec[i] + step >= matrix_data[i].shape[0]:
                return vec[0:i] + [matrix_data[i].shape[0] - 1] + vec[i + 1:]
            # 向上变异,移动step个单位
            return vec[0:i] + [vec[i] + step] + vec[i + 1:]

    def cross_over(r1, r2):
        """
        交叉算子
        :param r1:
        :param r2:
        :return:
        """
        m = np.random.randint(1, len(r1) - 2)
        return r1[:m] + r2[m:]

    # 随机生成初始种群
    population = generate_population(matrix_data, population_size)

    # 精英数
    top_elite_count = int(elite_rate * population_size)

    # 主循环迭代
    for i in range(max_iter):
        # 获取种群中所有基因的适应度和基因的元组
        scores = [(gene, adaptive_function(matrix_data, gene)) for gene in population]
        scores.sort(reverse=True, key=operator.itemgetter(1))
        print(scores[0], i + 1)
        # print(str(i) + '-------------------')
        # for score in scores:
        #     print(score)

        ranked_population = [g for (g, s) in scores]

        # 获取精英基因,淘汰劣质染色体
        population = ranked_population[:top_elite_count]

        while len(population) < population_size:
            if np.random.random() < mutate_prob:
                # 变异
                i = np.random.randint(0, top_elite_count)
                population.append(mutate(ranked_population[i]))
            else:
                # 交叉
                i = np.random.randint(0, top_elite_count)
                k = np.random.randint(0, top_elite_count)
                population.append(cross_over(ranked_population[i], ranked_population[k]))

    # 返回收敛后的解元组(适应度,染色体)
    return scores[0]


def random_optimize(matrix_data, adaptive_function, times):
    """
    随机搜索算法
    :param matrix_data:
    :param adaptive_function:
    :param times:
    :return:
    """
    vec = []
    for i in range(len(matrix_data)):
        vec.append(0)
    v_best = adaptive_function(matrix_data, vec)
    chromosome_best = vec[:]
    # 获取随机点
    population = generate_population(matrix_data, times)
    for chromosome in population:
        v_cur = adaptive_function(matrix_data, chromosome)
        if v_best < v_cur:
            v_best = v_cur
            chromosome_best = chromosome[:]
    return chromosome_best, v_best


def hill_climbing(matrix_data, adaptive_function, vec):
    """
    爬山算法
    :param matrix_data:
    :param adaptive_function:
    :param vec:
    :return:
    """
    while True:
        neighbours = []
        for i in range(len(vec)):
            if vec[i] + 1 < matrix_data[i].shape[0]:
                neighbours.append(vec[:i] + [vec[i] + 1] + vec[i + 1:])
            if vec[i] - 1 > 0:
                neighbours.append(vec[:i] + [vec[i] - 1] + vec[i + 1:])
        v_best = adaptive_function(matrix_data, vec)
        # vec_best = vec
        for neighbour in neighbours:
            v_cur = adaptive_function(matrix_data, neighbour)
            if v_best < v_cur:
                v_best = v_cur
                # vec_best = neighbour
        return v_best


def random_hill_climbing(matrix_data, adaptive_function, times, step=1):
    """
    随机重复爬山算法
    :param matrix_data:
    :param adaptive_function:
    :param times:
    :return:
    """
    # 获取初始随机点
    population = generate_population(matrix_data, times)
    best_chromosome = []
    for i in range(len(matrix_data)):
        best_chromosome.append(0)
    best_value = adaptive_function(matrix_data, best_chromosome)
    for chromosome in population:
        cur_value = hill_climbing(matrix_data, adaptive_function, chromosome)
        if cur_value > best_value:
            best_value = cur_value
            best_chromosome = chromosome
    return best_chromosome, best_value


def simulated_annealing(matrix_data, adaptive_function, temperature=10000, t_threshold=0.1, cooling_rate=0.1, step=1):
    """
    模拟退火算法
    :param matrix_data:
    :param adaptive_function:
    :param temperature:
    :param t_threshold:
    :param cooling_rate:
    :param step:
    :return:
    """
    vec = generate_population(matrix_data, 1)[0]
    v_origin = 0
    while temperature > t_threshold:
        # 选择一个索引位置
        i = np.random.randint(0, len(vec) - 1)
        # 选择一个改变索引的方向
        direction = np.random.randint(-step, step)
        # 创建新解
        vec_b = vec[:]
        if vec_b[i] + direction >= matrix_data[i].shape[0]:
            vec_b[i] = matrix_data[i].shape[0]
        elif vec_b[i] + direction < 0:
            vec_b[i] = 0
        else:
            vec_b[i] += direction

        # 当前解的目标函数
        v_origin = adaptive_function(matrix_data, vec)
        # 新解的目标函数
        v_b = adaptive_function(matrix_data, vec_b)

        # 接受更好的解,或者温度高时有一定概率接受更差的解
        if v_origin < v_b or np.random.random() < pow(math.e, -(v_b - v_origin) / temperature):
            vec = vec_b

        # 降温
        temperature *= cooling_rate
    return vec, v_origin


def random_sa(matrix_data, adaptive_function, times=1000, temperature=10000, t_threshold=0.1, cooling_rate=0.1, step=1):
    """
    随机重复模拟退火算法
    :param matrix_data:
    :param adaptive_function:
    :param times:
    :param temperature:
    :param t_threshold:
    :param cooling_rate:
    :param step:
    :return:
    """
    result = (None, 0)
    if times <= 0:
        times = 1000
    for i in range(times):
        cur_result = simulated_annealing(matrix_data, adaptive_function, temperature, t_threshold, cooling_rate, step)
        if cur_result[1] > result[1]:
            result = cur_result
    return result


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
    return population


if __name__ == '__main__':
    simulation_data = []
    for j in range(data.class_length):
        simulation_data.append(pd.read_csv(data.dt_path(j), index_col=0))

    # file_handler = open('test.data', 'w')
    # file_handler.write(str(simulation_data))
    # file_handler.close()

    # print(best_qos(simulation_data))

    # print('generic algorithm')
    # genetic_optimize(simulation_data, qos_total, max_iter=100, population_size=1000,
    #                  mutate_prob=0.95, step=4)

    # print('random search')
    # print(random_optimize(simulation_data, qos_total, 10000))

    # print('random restart hill climbing')
    # print(random_hill_climbing(simulation_data, qos_total, 10000))

    # print('simulated annealing')
    # print(simulated_annealing(simulation_data, qos_total))

    print('random restart simulated annealing')
    print(random_sa(simulation_data, qos_total))

    # test_gene = [4, 14, 18, 6, 12, 7]
    # print(qos_time(simulation_data, test_gene))
    # print(qos_cost(simulation_data, test_gene))
    # print(qos_availability(simulation_data, test_gene))
    # print(qos_reliability(simulation_data, test_gene))
    # print(qos_reputation(simulation_data, test_gene))
    # print(round(qos_total(simulation_data, test_gene), ndigits=4))
