# -*- coding: utf-8 -*-
import random
import pandas as pd
import data


def improved_abc(matrix_data, cost_function, onlooker_number, limit, max_iter=100):
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

    # 各个维度上取值的最大值和最小值
    df_data = pd.DataFrame(matrix_data)
    v_range = [df_data.min(), df_data.max()]

    # 蜜源
    nectar_source = data.get_population()
    # 蜜源未更新的累积迭代次数,超过limit次的话将被重新生成的蜜源代替
    iter_times = []
    # 初始化iter_times
    for i in range(nectar_source):
        iter_times.append(1)

    def generate_solution(min_, max_):
        """
        生成某个解的某个维度上的一个随机数,
        初始化时或者侦查蜂寻找新的蜜源时用
        :param min_: 解空间最小值
        :param max_: 解空间最大值
        :return:
        """
        return min_ + int(random.random() * (max_ - min_))

    def update_info(vec_):
        """
        更新蜜源信息
        :param vec_: 待更新蜜源
        :return:
        """
        k = int(random.random() * len(matrix_data))
        return [(vec_[index_] + int(random.random() * (v2 - vec_[index_]) + v_range[1][index_])) % v_range[1][index_]
                for index_, v2 in enumerate(matrix_data[k])]

    def update_prob_territory():
        """
        更新适应度百分概率分布信息
        :return:
        """
        probability_territory_ = []
        fitness_sum = 0
        for vec_ in nectar_source:
            fitness_sum += cost_function(vec_)
        current_fitness_sum = 0
        for vec_ in nectar_source:
            current_fitness_sum += cost_function(vec_)
            probability_territory_.append(float(current_fitness_sum) / fitness_sum)
        return probability_territory_

    def search_nectar(vec_old_, vec_new_):
        """
        一次搜索,确定两个蜜源的蜜量,返回较大蜜量的蜜源
        :param vec_old_:
        :param vec_new_:
        :return: 更优蜜源,蜜源蜜量,蜜源是否更新过
        """
        nectar_quantity_old_ = cost_function(vec_old_)
        nectar_quantity_new_ = cost_function(vec_new_)

        if nectar_quantity_new_ > nectar_quantity_old_:
            return vec_new_, nectar_quantity_new_, True
        return vec_old_, nectar_quantity_old_, False

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
            rand_num = random.random()
            index = len(probability_territory) - 1
            for j, pt in enumerate(probability_territory):
                if rand_num > pt:
                    index = j - 1
                    break
            # 搜索第index个蜜源区域
            nectar_new = update_info(vec)
            nectar_source[index], nectar_quantity_temp, changed = search_nectar(vec, nectar_new)
            # 记录蜜源不变的迭代次数
            if changed:
                iter_times[i] = 0
            else:
                iter_times[i] += 1

        # 采蜜蜂是否需要成为侦查蜂重新侦查新的蜜源,根据蜜源不变的迭代次数和limit参数确实定否侦查
        for i in iter_times:
            if i >= limit:
                vec_new = []
                for j in range(nectar_source[0]):
                    vec_new.append(generate_solution(v_range[0][j], v_range[1][j]))
                nectar_source[i] = vec_new

    # 获取最佳蜜源
    nectar_best = cost_function(nectar_source[0])
    for vec in nectar_source:
        nectar_temp = cost_function(vec)
        if nectar_best < nectar_temp:
            nectar_best = nectar_temp
    return nectar_best


if __name__ == '__main__':
    mt = data.get_population()
    print(mt)
    improved_abc(mt, 1, 1, 1, 1)
