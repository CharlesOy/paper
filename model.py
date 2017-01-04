# -*- coding: utf-8 -*-
import time
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
    result = 0.1 * qos_time(matrix_data, gene)
    result += 0.1 * qos_cost(matrix_data, gene)
    result += 0.3 * qos_availability(matrix_data, gene)
    result += 0.3 * qos_reliability(matrix_data, gene)
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


def random_optimize(matrix_data, adaptive_function):
    """
    随机搜索算法
    :param matrix_data:
    :param adaptive_function:
    :return:
    """
    # 记录算法开始时间戳
    t_begin = time.time()
    vec = []
    for i in range(len(matrix_data)):
        vec.append(0)
    v_best = adaptive_function(matrix_data, vec)
    chromosome_best = vec[:]
    # 获取随机点
    population = data.get_population()
    for chromosome in population:
        v_cur = adaptive_function(matrix_data, chromosome)
        if v_best < v_cur:
            v_best = v_cur
            chromosome_best = chromosome[:]
    # 返回元组为(解,值,运行时间)
    return chromosome_best, v_best, time.time() - t_begin


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
        for k in range(len(vec)):
            if vec[k] + 1 < matrix_data[k].shape[0]:
                neighbours.append(vec[:k] + [vec[k] + 1] + vec[k + 1:])
            if vec[k] - 1 > 0:
                neighbours.append(vec[:k] + [vec[k] - 1] + vec[k + 1:])
        v_best = adaptive_function(matrix_data, vec)
        # vec_best = vec
        for neighbour in neighbours:
            v_cur = adaptive_function(matrix_data, neighbour)
            if v_best < v_cur:
                v_best = v_cur
                # vec_best = neighbour
        return v_best


def random_hill_climbing(matrix_data, adaptive_function):
    """
    随机重复爬山算法
    :param matrix_data:
    :param adaptive_function:
    :return:
    """
    # 记录算法开始时间
    t_begin = time.time()
    # 获取初始随机点
    population = data.get_population()
    best_chromosome = []
    for i in range(len(matrix_data)):
        best_chromosome.append(0)
    best_value = adaptive_function(matrix_data, best_chromosome)
    for chromosome in population:
        cur_value = hill_climbing(matrix_data, adaptive_function, chromosome)
        if cur_value > best_value:
            best_value = cur_value
            best_chromosome = chromosome
    # 返回的元组为(解,值,运行时间)
    return best_chromosome, best_value, time.time() - t_begin


def random_simulated_annealing(matrix_data, adaptive_function, temperature=10000, t_threshold=0.1, cooling_rate=0.1,
                               step=1):
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
    # 记录算法开始时间戳
    t_begin = time.time()
    # 获取起始种群
    population = data.get_population()
    best_vec = population[0]
    best_value = adaptive_function(matrix_data, best_vec)
    for vec in population:
        v_origin = 0
        cur_temperature = temperature
        while cur_temperature > t_threshold:
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
            if v_origin < v_b or np.random.random() < pow(math.e, -(v_b - v_origin) / cur_temperature):
                vec = vec_b

            # 降温
            cur_temperature *= cooling_rate
        # 记录最佳值和解
        if v_origin > best_value:
            best_value = v_origin
            best_vec = vec
    # 返回元组为(解,值,运行时间)
    return best_vec, best_value, time.time() - t_begin


def ga(matrix_data, adaptive_function, mutate_prob=0.2, step=1, elite_rate=0.2,
       max_iter=100):
    """
    经典遗传算法,
    采用选择算子采用排序法,
    变异算子,
    交叉算子采用随机交叉
    :param matrix_data:
    :param adaptive_function:
    :param mutate_prob:
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
        m = np.random.randint(0, len(matrix_data))
        if np.random.random() < 0.5:
            # 染色体变异位置超过下限
            if vec[m] - step < 0:
                return vec[0:m] + [0] + vec[m + 1:]
            # 向下变异,移动step个单位
            return vec[0:m] + [vec[m] - step] + vec[m + 1:]
        else:
            # 染色体变异位置超过上限
            if vec[m] + step >= matrix_data[m].shape[0]:
                return vec[0:m] + [matrix_data[m].shape[0] - 1] + vec[m + 1:]
            # 向上变异,移动step个单位
            return vec[0:m] + [vec[m] + step] + vec[m + 1:]

    def cross_over(r1, r2):
        """
        交叉算子
        :param r1:
        :param r2:
        :return:
        """
        m = np.random.randint(1, len(r1) - 2)
        return r1[:m] + r2[m:]

    # 记录算法开始时间戳
    t_begin = time.time()
    # 记录迭代过程
    procedure = []

    # 随机生成初始种群
    population = data.get_population()
    population_size = len(population)

    # 精英数
    top_elite_count = int(elite_rate * population_size)

    # 主循环迭代
    for i in range(max_iter):
        # 获取种群中所有基因的适应度和基因的元组
        scores = [(gene, adaptive_function(matrix_data, gene)) for gene in population]
        scores.sort(reverse=True, key=operator.itemgetter(1))
        procedure.append((scores[0], i + 1, time.time() - t_begin))
        # print(scores[0], i + 1)
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
    return procedure


def iga():
    """

    :return:
    """
    return 'nothing'


def improved_pso(matrix_data, adaptive_function, max_iter=100, weight=0.8, c1=2, c2=2):
    """
    一种改进的粒子群优化算法
    :param matrix_data:
    :param adaptive_function:
    :param max_iter:
    :param weight:
    :param c1:
    :param c2:
    :return:
    """
    # 算法开始时间戳
    t_begin = time.time()
    # 记录迭代状态
    procedure = []
    # 粒子群
    population = data.get_population()
    # 随机速度
    speeds = data.generate_population(matrix_data, len(population))
    # 每个粒子的最佳纪录
    best_histories = []
    # 粒子群的最佳纪录
    best_of_all = (None, 0)
    # 初始化每个粒子的最佳纪录
    for chromosome in population:
        value_cur = adaptive_function(matrix_data, chromosome)
        best_histories.append((chromosome, value_cur))
        if value_cur > best_of_all[1]:
            best_of_all = (chromosome, value_cur)
    # 主循环
    for i in range(max_iter):
        procedure.append((best_of_all, i + 1, time.time() - t_begin))
        # print(best_of_all, i)
        # print(best_histories[0])
        # print(population[0])
        # print(speeds[0])
        # print('')
        for k in range(len(population)):
            chromosome = population[k]
            speed = speeds[k]
            for m in range(len(chromosome)):
                # 计算速度
                speed[m] *= speed[m] * weight
                speed[m] += c1 * np.random.random() * (best_histories[k][0][m] - chromosome[m])
                speed[m] += c2 * np.random.random() * (best_of_all[0][m] - chromosome[m])
                speed[m] = int(round(speed[m])) % matrix_data[m].shape[0]
                # 计算新的位置
                chromosome[m] += speed[m]
                chromosome[m] %= matrix_data[m].shape[0]

                # if chromosome[m] + speed[m] >= matrix_data[m].shape[0]:
                #     chromosome[m] = matrix_data[m].shape[0] - 1
                # elif chromosome[m] + speed[m] < 0:
                #     chromosome[m] = 0
                # else:
                #     chromosome[m] += speed[m]
            value_cur = adaptive_function(matrix_data, chromosome)
            if value_cur > best_histories[k][1]:
                best_histories[k] = (chromosome, value_cur)
            if value_cur > best_of_all[1]:
                # print('updated')
                best_of_all = (chromosome[:], value_cur)

    procedure.append((best_of_all, i + 1, time.time() - t_begin))

    return procedure


def improved_ga(matrix_data, adaptive_function, chromosome_mutate_rate=0.2, step=1, elite_rate=0.2,
                max_iter=100, temperature=10000, threshold_mutate_prob=0.1, cooling_rate=0.8):
    """
    变异算子经过改进的遗传算法,
    选择算子采用模拟退火思想,
    变异算子遍历选取局部最优解,
    交叉算子采用随机交叉
    :param matrix_data:目标矩阵
    :param adaptive_function:适应度函数
    :param chromosome_mutate_rate: 每次变异时,染色体中有chromosome_mutate_rate的比例的基因将会发生突变
    :param step:单个基因变异移动步骤
    :param elite_rate:精英留存率
    :param max_iter:最大迭代次数
    :param temperature:温度
    :param threshold_mutate_prob:最高变异率
    :param cooling_rate:冷却率
    :return:
    """

    # def mutate(vec):
    #     """
    #     变异算子
    #     :param vec:染色体
    #     :return:
    #     """
    #     m = np.random.randint(0, len(matrix_data))
    #     cur_vec = vec[0:m] + [(vec[m] - step + matrix_data[m].shape[0]) % matrix_data[m].shape[0]] + vec[m + 1:]
    #     best_val = adaptive_function(matrix_data, cur_vec)
    #     # temp = best_val
    #     best_vec = cur_vec
    #     for n in range(int(chromosome_mutate_rate * len(vec))):
    #         m = np.random.randint(0, len(matrix_data))
    #         if np.random.random() < 0.5:
    #             # 向下变异,移动step个单位
    #             cur_vec = vec[0:m] + [(vec[m] - step + matrix_data[m].shape[0]) % matrix_data[m].shape[0]] + vec[m + 1:]
    #         else:
    #             # 向上变异,移动step个单位
    #             cur_vec = vec[0:m] + [(vec[m] + step) % matrix_data[m].shape[0]] + vec[m + 1:]
    #         cur_val = adaptive_function(matrix_data, cur_vec)
    #         if cur_val > best_val:
    #             best_val = cur_val
    #             best_vec = cur_vec[:]
    #     # print(temp, best_val)
    #     return best_vec

    # def mutate2(vec):
    #     """
    #     终极大杀器,
    #     当前模型不好,
    #     基因之间不相关,
    #     每个基因都有自己的最佳编码,
    #     因此只要获取染色体的各个基因位置各自的最佳编码,
    #     该染色体就是最佳染色体(最优解),
    #     嘛,写论文的时候我是不会写进去的...哈哈...
    #     改进过的变异算子
    #     随机基因位置,
    #     确定该基因最合适编码
    #     :param vec:
    #     :return:
    #     """
    #     m = np.random.randint(0, len(matrix_data))
    #     best_chromosome = 0
    #     best_v = 0
    #     for index in range(0, matrix_data[m].shape[0]):
    #         cur_v = adaptive_function(matrix_data, vec[0:m] + [index] + vec[m + 1:])
    #         if cur_v > best_v:
    #             best_v = cur_v
    #             best_chromosome = vec[0:m] + [index] + vec[m + 1:]
    #     return best_chromosome

    def mutate(vec, accept_prob=0):
        """
        改进过的变异算子,
        随机基因位置,
        确定该基因最合适编码
        :param vec:
        :param accept_prob:
        :return:
        """
        m = np.random.randint(0, len(matrix_data))
        best_val = 0
        g_index = np.random.randint(0, matrix_data[m].shape[0])
        best_vec = None
        while True:
            cur_vec = vec[0:m] + [g_index] + vec[m + 1:]
            cur_val = adaptive_function(matrix_data, cur_vec)
            if best_val < cur_val or np.random.random() < accept_prob:
                best_val = cur_val
                best_vec = cur_vec[:]
            else:
                break
        return best_vec

    # def mutate3(vec):
    #     """
    #     改进过的变异算子,
    #     随机基因位置,
    #     确定该基因最合适编码
    #     :param vec:
    #     :return:
    #     """
    #     m = np.random.randint(0, len(matrix_data))
    #     best_val = 0
    #     g_index = np.random.randint(0, matrix_data[m].shape[0])
    #     best_vec = None
    #     while True:
    #         cur_vec = vec[0:m] + [g_index] + vec[m + 1:]
    #         cur_val = adaptive_function(matrix_data, cur_vec)
    #         if best_val < cur_val:
    #             best_val = cur_val
    #             best_vec = cur_vec[:]
    #         else:
    #             break
    #     return best_vec

    def cross_over(r1, r2):
        """
        交叉算子
        :param r1:
        :param r2:
        :return:
        """
        result = []
        for m in range(len(r1)):
            if np.random.random() > 0.5:
                result.append(r1[m])
            else:
                result.append(r2[m])
        return result

    # 记录算法开始时间戳
    t_begin = time.time()

    # 记录每一代的最优状态
    procedure = []

    # 随机生成初始种群
    population = data.get_population()
    population_size = len(population)

    # 精英数
    top_elite_count = int(elite_rate * population_size)

    # 主循环迭代
    for i in range(max_iter):
        # 获取种群中所有基因的适应度和基因的元组
        scores = [(gene, adaptive_function(matrix_data, gene)) for gene in population]
        scores.sort(reverse=True, key=operator.itemgetter(1))
        procedure.append((scores[0], i + 1, time.time() - t_begin))
        # print(scores[0], i + 1)
        # print(str(i) + '-------------------')
        # for score in scores:
        #     print(score)

        ranked_population = [g for (g, v) in scores]

        # 动态计算变异率,模拟退火算法计算变异率,如得到的变异率大于设定的最低变异率,则按最低变异率按最大变异率计算
        population = ranked_population[:top_elite_count]
        mutate_prob = 1 - pow(math.e, -(scores[top_elite_count - 1][1] - scores[0][1])) / temperature
        if threshold_mutate_prob > mutate_prob:
            mutate_prob = threshold_mutate_prob
        else:
            # 降温
            temperature *= cooling_rate
        # 获取精英基因,淘汰劣质染色体
        while len(population) < population_size:
            if np.random.random() < mutate_prob:
                # 变异
                i = np.random.randint(0, top_elite_count)
                population.append(mutate(ranked_population[i], 1 - mutate_prob))
            else:
                # 交叉
                i = np.random.randint(0, top_elite_count)
                k = np.random.randint(0, top_elite_count)
                population.append(cross_over(ranked_population[i], ranked_population[k]))

                # 返回收敛后的解元组(适应度,染色体)
    # 返回的元组((解,值),代数,运行时间)
    return procedure


def hybrid_ga(matrix_data, adaptive_function, chromosome_mutate_rate=0.2, step=1, elite_rate=0.2,
              max_iter=100, temperature=10000, threshold_mutate_prob=0.1, cooling_rate=0.8):
    """
    变异算子经过改进的遗传算法,
    选择算子采用模拟退火思想,
    变异算子遍历选取局部最优解,
    交叉算子采用随机交叉
    :param matrix_data:目标矩阵
    :param adaptive_function:适应度函数
    :param chromosome_mutate_rate: 每次变异时,染色体中有chromosome_mutate_rate的比例的基因将会发生突变
    :param step:单个基因变异移动步骤
    :param elite_rate:精英留存率
    :param max_iter:最大迭代次数
    :param temperature:温度
    :param threshold_mutate_prob:最高变异率
    :param cooling_rate:冷却率
    :return:
    """

    # def mutate(vec):
    #     """
    #     变异算子
    #     :param vec:染色体
    #     :return:
    #     """
    #     m = np.random.randint(0, len(matrix_data))
    #     cur_vec = vec[0:m] + [(vec[m] - step + matrix_data[m].shape[0]) % matrix_data[m].shape[0]] + vec[m + 1:]
    #     best_val = adaptive_function(matrix_data, cur_vec)
    #     # temp = best_val
    #     best_vec = cur_vec
    #     for n in range(int(chromosome_mutate_rate * len(vec))):
    #         m = np.random.randint(0, len(matrix_data))
    #         if np.random.random() < 0.5:
    #             # 向下变异,移动step个单位
    #             cur_vec = vec[0:m] + [(vec[m] - step + matrix_data[m].shape[0]) % matrix_data[m].shape[0]] + vec[m + 1:]
    #         else:
    #             # 向上变异,移动step个单位
    #             cur_vec = vec[0:m] + [(vec[m] + step) % matrix_data[m].shape[0]] + vec[m + 1:]
    #         cur_val = adaptive_function(matrix_data, cur_vec)
    #         if cur_val > best_val:
    #             best_val = cur_val
    #             best_vec = cur_vec[:]
    #     # print(temp, best_val)
    #     return best_vec

    # def mutate2(vec):
    #     """
    #     终极大杀器,
    #     当前模型不好,
    #     基因之间不相关,
    #     每个基因都有自己的最佳编码,
    #     因此只要获取染色体的各个基因位置各自的最佳编码,
    #     该染色体就是最佳染色体(最优解),
    #     嘛,写论文的时候我是不会写进去的...哈哈...
    #     改进过的变异算子
    #     随机基因位置,
    #     确定该基因最合适编码
    #     :param vec:
    #     :return:
    #     """
    #     m = np.random.randint(0, len(matrix_data))
    #     best_chromosome = 0
    #     best_v = 0
    #     for index in range(0, matrix_data[m].shape[0]):
    #         cur_v = adaptive_function(matrix_data, vec[0:m] + [index] + vec[m + 1:])
    #         if cur_v > best_v:
    #             best_v = cur_v
    #             best_chromosome = vec[0:m] + [index] + vec[m + 1:]
    #     return best_chromosome

    def mutate(vec, accept_prob=0):
        """
        改进过的变异算子,
        随机基因位置,
        确定该基因最合适编码
        :param vec:
        :param accept_prob:
        :return:
        """
        m = np.random.randint(0, len(matrix_data))
        best_val = 0
        g_index = np.random.randint(0, matrix_data[m].shape[0])
        best_vec = None
        while True:
            cur_vec = vec[0:m] + [g_index] + vec[m + 1:]
            cur_val = adaptive_function(matrix_data, cur_vec)
            if best_val < cur_val or np.random.random() < accept_prob:
                best_val = cur_val
                best_vec = cur_vec[:]
            else:
                break
        return best_vec

    # def mutate3(vec):
    #     """
    #     改进过的变异算子,
    #     随机基因位置,
    #     确定该基因最合适编码
    #     :param vec:
    #     :return:
    #     """
    #     m = np.random.randint(0, len(matrix_data))
    #     best_val = 0
    #     g_index = np.random.randint(0, matrix_data[m].shape[0])
    #     best_vec = None
    #     while True:
    #         cur_vec = vec[0:m] + [g_index] + vec[m + 1:]
    #         cur_val = adaptive_function(matrix_data, cur_vec)
    #         if best_val < cur_val:
    #             best_val = cur_val
    #             best_vec = cur_vec[:]
    #         else:
    #             break
    #     return best_vec

    def cross_over(r1, r2):
        """
        交叉算子
        :param r1:
        :param r2:
        :return:
        """
        result = []
        for m in range(len(r1)):
            if np.random.random() > 0.5:
                result.append(r1[m])
            else:
                result.append(r2[m])
        return result

    # 记录算法开始时间戳
    t_begin = time.time()

    # 记录每一代的最优状态
    procedure = []

    # 随机生成初始种群
    population = data.get_population()
    population_size = len(population)

    # 精英数
    top_elite_count = int(elite_rate * population_size)

    # 主循环迭代
    for i in range(max_iter):
        # 获取种群中所有基因的适应度和基因的元组
        scores = [(gene, adaptive_function(matrix_data, gene)) for gene in population]
        scores.sort(reverse=True, key=operator.itemgetter(1))
        procedure.append((scores[0], i + 1, time.time() - t_begin))
        # print(scores[0], i + 1)
        # print(str(i) + '-------------------')
        # for score in scores:
        #     print(score)

        ranked_population = [g for (g, v) in scores]

        # 动态计算变异率,模拟退火算法计算变异率,如得到的变异率大于设定的最低变异率,则按最低变异率按最大变异率计算
        population = ranked_population[:top_elite_count]
        mutate_prob = 1 - pow(math.e, -(scores[top_elite_count - 1][1] - scores[0][1])) / temperature
        if threshold_mutate_prob > mutate_prob:
            mutate_prob = threshold_mutate_prob
        else:
            # 降温
            temperature *= cooling_rate
        # 获取精英基因,淘汰劣质染色体
        while len(population) < population_size:
            if np.random.random() < mutate_prob:
                # 变异
                i = np.random.randint(0, top_elite_count)
                population.append(mutate(ranked_population[i], 1 - mutate_prob))
            else:
                # 交叉
                i = np.random.randint(0, top_elite_count)
                k = np.random.randint(0, top_elite_count)
                population.append(cross_over(ranked_population[i], ranked_population[k]))

                # 返回收敛后的解元组(适应度,染色体)
    # 返回的元组((解,值),代数,运行时间)
    return procedure


def improved_iga(matrix_data, adaptive_function, mutate_prob=0.2, step=1, elite_rate=0.2,
                            max_iter=100, alpha=0.2, beta=0.8, gate=0.3):
    """
    免疫遗传算法,
    采用选择算子采用排序法,
    变异算子,
    交叉算子采用随机交叉
    :param matrix_data:
    :param adaptive_function:
    :param mutate_prob:
    :param step:
    :param elite_rate:
    :param max_iter:
    :param alpha: 疫苗注射概率
    :param beta: 局部最优的概率阈值,如果beta个种群的适应度和种群的平均适应度的绝对值小于gate,则表示需要接种疫苗
    :param gate: 判断种群是否进入局部优化的指标
    :return:
    """

    def mutate(vec):
        """
        变异算子
        :param vec:
        :return:
        """
        m = np.random.randint(0, len(matrix_data))
        if np.random.random() < 0.5:
            # 染色体变异位置超过下限
            if vec[m] - step < 0:
                return vec[0:m] + [0] + vec[m + 1:]
            # 向下变异,移动step个单位
            return vec[0:m] + [vec[m] - step] + vec[m + 1:]
        else:
            # 染色体变异位置超过上限
            if vec[m] + step >= matrix_data[m].shape[0]:
                return vec[0:m] + [matrix_data[m].shape[0] - 1] + vec[m + 1:]
            # 向上变异,移动step个单位
            return vec[0:m] + [vec[m] + step] + vec[m + 1:]

    def cross_over(r1, r2):
        """
        交叉算子
        :param r1:
        :param r2:
        :return:
        """
        m = np.random.randint(1, len(r1) - 2)
        return r1[:m] + r2[m:]

    # 疫苗库
    vaccines = []

    # 记录算法开始时间戳
    t_begin = time.time()

    # 记录迭代过程
    procedure = []

    # 随机生成初始种群
    population = data.get_population()
    population_size = len(population)

    # 疫苗库长度
    vaccines_length = population_size + 20

    # 精英数
    top_elite_count = int(elite_rate * population_size)

    # 注射疫苗时,原种群的保留数目
    reservation_count = population_size * (1 - alpha)

    def add_vaccine(vecs):
        """
        新增疫苗到疫苗库,如果超过疫苗库的限制长度,则优先保存适应度高的疫苗
        :param vecs:待新增的疫苗
        :return:
        """
        for vec in vecs:
            if len(vaccines) == 0:
                vaccines.append(vec)
            elif len(vaccines) > vaccines_length:
                vaccines.pop()
                for _i in range(len(vaccines)):
                    vaccine = vaccines[_i]
                    if adaptive_function(matrix_data, vaccine) < adaptive_function(matrix_data, vec):
                        vaccines.insert(_i, vec)
                        break
            else:
                for _i in range(len(vaccines)):
                    vaccine = vaccines[_i]
                    if adaptive_function(matrix_data, vaccine) < adaptive_function(matrix_data, vec):
                        vaccines.insert(_i, vec)
                        break

    def insert_vaccine():
        """
        注射alpha比例的疫苗到种群中
        :return:
        """
        temp_vaccines = []
        for _i in range(population_size * alpha):
            temp_vaccines.append(vaccines[_i])
        return temp_vaccines

    def local_best():
        """
        判断是否陷入局部最优
        :return:
        """
        average_fit = 0
        for chromosome in population:
            average_fit += adaptive_function(matrix_data, chromosome)
        average_fit /= len(population)

        similarity_count = 0
        for chromosome in population:
            if math.fabs(adaptive_function(matrix_data, chromosome) - average_fit) < gate:
                similarity_count += 1

        return similarity_count > beta * population_size

    # 主循环迭代
    for i in range(max_iter):
        # 获取种群中所有基因的适应度和基因的元组
        scores = [(gene, adaptive_function(matrix_data, gene)) for gene in population]
        scores.sort(reverse=True, key=operator.itemgetter(1))
        procedure.append((scores[0], i + 1, time.time() - t_begin))
        # print(scores[0], i + 1)
        # print(str(i) + '-------------------')
        # for score in scores:
        #     print(score)

        ranked_population = [g for (g, s) in scores]

        # 添加最优染色体到疫苗库
        # 这里暂定只添加一个
        add_vaccine([ranked_population[0]])

        # 获取精英基因,淘汰劣质染色体
        population = ranked_population[:top_elite_count]

        # 如果陷入局部最优,则注射疫苗
        # print(i)
        if local_best():
            print(str(i) + "test local best")
            population = population[:reservation_count] + insert_vaccine()
        else:
            pass

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
    return procedure


def improved_abc(matrix_data, cost_function, onlooker_number=500, limit=10, max_iter=50):
    """
    一种改进蜂群算法(在本文中用于对比),
    蜜源和采蜜蜂的数量有data.get_population()方法决定
    :param matrix_data: 搜索空间
    :param cost_function: 目标函数
    :param onlooker_number: 观察蜂个数
    :param limit: 单个蜜源搜索限制次数
    :param max_iter: 最大迭代次数
    :return:
    """

    """
    # 简化人工蜂群的解释过程
    1.随机生成一定数量的解
    2.对于每一次循环有:
    3.  每个解的每个维度取值按照公式一变动一次,如果适应度更优,则更新解
    4.  按照适应度的比例,随机更新每个解,总次数一定
    5.  如果超过limit次迭代解都没有得到更新,则使用一个新的随机解代替该解
    6.达到循环结束条件
    """

    # 记录算法开始时间戳
    t_begin = time.time()

    # 记录每一代的最优状态
    procedure = []

    # 蜜源
    nectar_source = data.get_population()

    # 各个维度上取值的最大值和最小值
    df_data = pd.DataFrame(nectar_source)
    v_range = [df_data.min(), df_data.max()]

    # 蜜源未更新的累积迭代次数,超过limit次的话将被重新生成的蜜源代替
    iter_times = []
    # 初始化iter_times
    for i in range(len(nectar_source)):
        iter_times.append(1)

    def generate_solution(min_, max_):
        """
        生成某个解的某个维度上的一个随机数,
        初始化时或者侦查蜂寻找新的蜜源时用
        :param min_: 解空间最小值
        :param max_: 解空间最大值
        :return:
        """
        return min_ + int(np.random.random() * (max_ - min_))

    def update_info(vec_):
        """
        更新蜜源信息
        :param vec_: 待更新蜜源
        :return:
        """
        k = int(np.random.random() * len(nectar_source))
        return [
            (vec_[index__] + int(np.random.random() * (v2 - vec_[index__]) + v_range[1][index__])) % v_range[1][index__]
            for index__, v2 in enumerate(nectar_source[k])]

    def update_prob_territory():
        """
        更新适应度百分概率分布信息
        :return:
        """
        probability_territory_ = []
        fitness_sum = 0
        for vec_ in nectar_source:
            fitness_sum += cost_function(matrix_data, vec_)
        current_fitness_sum = 0
        for vec_ in nectar_source:
            current_fitness_sum += cost_function(matrix_data, vec_)
            probability_territory_.append(float(current_fitness_sum) / fitness_sum)
        return probability_territory_

    def search_nectar(vec_old_, vec_new_):
        """
        一次搜索,确定两个蜜源的蜜量,返回较大蜜量的蜜源
        :param vec_old_:
        :param vec_new_:
        :return: 更优蜜源,蜜源蜜量,蜜源是否更新过
        """
        nectar_quantity_old_ = cost_function(matrix_data, vec_old_)
        nectar_quantity_new_ = cost_function(matrix_data, vec_new_)

        if nectar_quantity_new_ > nectar_quantity_old_:
            return vec_new_, nectar_quantity_new_, True
        return vec_old_, nectar_quantity_old_, False

    nectar_best = nectar_source[0]
    for it in range(max_iter):
        # 采蜜蜂随机寻找新的蜜源,贪婪思想更新蜜源(设置含蜜量更多的蜜源为新蜜源)
        for i in range(len(nectar_source)):
            vec = nectar_source[i]
            nectar_new = update_info(vec)
            nectar_source[i], nectar_quantity_temp, changed = search_nectar(vec, nectar_new)
            # 记录蜜源不变的迭代次数
            if changed:
                iter_times[i] = 0
            else:
                iter_times[i] += 1

        # 适应度百分概率条形图,记录从某个位置到某个位置
        probability_territory = update_prob_territory()

        # 观察蜂根据采蜜蜂更新的蜜源信息随机确定搜索的蜜源区域
        for i in range(onlooker_number):
            # 确定当前观察蜂搜索的蜜源区域
            rand_num = np.random.random()
            index = len(probability_territory) - 1
            for j in range(len(probability_territory)):
                pt = probability_territory[j]
                if rand_num < pt:
                    index = j
                    break
            # 搜索第index个蜜源区域
            vec = nectar_source[index]
            nectar_new = update_info(vec)
            nectar_source[index], nectar_quantity_temp, changed = search_nectar(vec, nectar_new)
            # 记录蜜源不变的迭代次数
            if changed:
                iter_times[index] = 0
            else:
                iter_times[index] += 1

        # 采蜜蜂是否需要成为侦查蜂重新侦查新的蜜源,根据蜜源不变的迭代次数和limit参数确实定否侦查
        for i in range(len(iter_times)):
            times = iter_times[i]
            if times >= limit and nectar_source[i] != nectar_best:
                vec_new = []
                for j in range(len(nectar_source[0])):
                    vec_new.append(generate_solution(v_range[0][j], v_range[1][j]))
                nectar_source[i] = vec_new

        # 获取最佳蜜源
        nectar_quantity_best = cost_function(matrix_data, nectar_source[0])
        nectar_best = None
        for vec in nectar_source:
            nectar_quantity_temp = cost_function(matrix_data, vec)
            if nectar_quantity_best < nectar_quantity_temp:
                nectar_quantity_best = nectar_quantity_temp
                nectar_best = vec
        # print(nectar_quantity_best, it + 1, time.time() - t_begin)
        procedure.append((nectar_quantity_best, it + 1, time.time() - t_begin))

    return procedure


def improved_quantum_ga():
    print('test')


if __name__ == '__main__':
    # 读取仿真数据矩阵的数组
    simulation_data = []
    for j_ in range(data.class_length):
        simulation_data.append(pd.read_csv(data.dt_path(j_), index_col=0))

    '''
    各个算法重复20次试验并保存结果

    每次实验分别运行6个算法
    随机搜索...1
    重复爬山...2
    重复模拟退火...3
    改进后的遗传算法(目标算法)...4
    遗传算法...5
    粒子群优化...6

    实验结果元组格式为(解,值,运行时间)
    其中算法4,5,6保存每一次迭代的状态

    每次算法1,2,3的初始随机解即群体智能算法(算法4,5,6)的初始种群
    此外每次实验的初始种群(初始随机解)重新生成一次
    '''
    # 实验总用时起始时间戳
    t_begin_ = time.time()
    # 实验的种群大小
    population_size_ = 200
    for index_ in range(1, 21):
        # # 生成初始种群(初始随机解)
        data.generate_population(simulation_data, population_size_)

        # # 随机搜索
        # print('random search ' + str(index_))
        # result_ = random_optimize(simulation_data, qos_total)
        # path = 'result/' + str(index_) + '_1_random_search.pkl'
        # data.write_result(path, result_)
        # # print(data.read_result(path))
        #
        # # 重复爬山
        # print('random restart hill climbing ' + str(index_))
        # result_ = random_hill_climbing(simulation_data, qos_total)
        # path = 'result/' + str(index_) + '_2_random_restart_hill_climbing.pkl'
        # data.write_result(path, result_)
        # # print(data.read_result(path))
        #
        # # 重复模拟退火
        # print('simulated annealing ' + str(index_))
        # result_ = random_simulated_annealing(simulation_data, qos_total)
        # path = 'result/' + str(index_) + '_3_simulated_annealing.pkl'
        # data.write_result(path, result_)
        # # print(data.read_result(path))
        #
        # # 遗传算法
        # print('genetic algorithm ' + str(index_))
        # result_ = ga(simulation_data, qos_total, max_iter=100, mutate_prob=0.95, step=4)
        # path = 'result/' + str(index_) + '_4_genetic_algorithm.pkl'
        # data.write_result(path, result_)
        # # data.print_array(data.read_result(path))

        # 免疫遗传算法
        print('immune genetic algorithm ' + str(index_))
        result_ = iga(simulation_data, qos_total, max_iter=100, mutate_prob=0.95, step=4)
        path = 'result/' + str(index_) + '_5_genetic_algorithm.pkl'
        data.write_result(path, result_)
        # data.print_array(data.read_result(path))

        # 混合免疫算法
        print('hybrid genetic algorithm ' + str(index_))
        result_ = hybrid_ga(simulation_data, qos_total, max_iter=100, threshold_mutate_prob=0.95)
        path = 'result/' + str(index_) + '_6_improved_genetic_algorithm.pkl'
        data.write_result(path, result_)
        data.print_array(data.read_result(path))

        # 改进遗传算法
        print('improved genetic algorithm ' + str(index_))
        result_ = improved_ga(simulation_data, qos_total, max_iter=100, threshold_mutate_prob=0.95)
        path = 'result/' + str(index_) + '_7_improved_genetic_algorithm.pkl'
        data.write_result(path, result_)
        data.print_array(data.read_result(path))

        # 改进免疫算法
        print('improved immune genetic algorithm ' + str(index_))
        result_ = improved_iga(simulation_data, qos_total, max_iter=100, mutate_prob=0.95, step=4)
        # print(result_)
        path = 'result/' + str(index_) + '_8_immune_genetic_algorithm.pkl'
        data.write_result(path, result_)
        # data.print_array(data.read_result(path))

        # 改进粒子群优化
        print('improved particle swarm optimization ' + str(index_))
        result_ = improved_pso(simulation_data, qos_total, max_iter=700)
        path = 'result/' + str(index_) + '_9_particle_swarm_optimization.pkl'
        data.write_result(path, result_)
        # data.print_array(data.read_result(path))

        # 改进人工蜂群算法
        print('improved artificial bee colony ' + str(index_))
        result_ = improved_abc(simulation_data, qos_total)
        # print(result_)
        path = 'result/' + str(index_) + '_10_artificial_bee_colony.pkl'
        data.write_result(path, result_)
        # data.print_array(data.read_result(path))

    # 总用时
    print(time.time() - t_begin_)
