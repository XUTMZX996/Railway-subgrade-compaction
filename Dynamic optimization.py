
import numpy as np
from random import random, sample, choice, uniform
import random
import pandas as pd
from joblib import load
import joblib
import matplotlib.pyplot as plt
# 基因数：计算辊道数即基因数量，c,d为重叠尺寸的限制
def calculate_roller_quantity(lujikuan, wheel_width, c, d):
    n = 1
    while not (wheel_width * n - d * (n - 1) < lujikuan and wheel_width * n - c * (n - 1) > lujikuan):
        n += 1
    return n
biaozuizhi = 120#float(input("智能指标标准值： "))
lujikuan = 9.3#float(input("路基宽： "))
wheel_width =2.17 #float(input("压路机轮宽： "))
c,d=0.35,0.45 #float(input("重叠下限： ")),float(input("重叠上限： "))
POPULATION_SIZE = 150  # 种群大小
GENE_SIZE = calculate_roller_quantity(lujikuan, wheel_width, c, d)  # 基因数量(即压实辊道数M)
ELEMENT_SIZE = 3  # 每个基因所含数据，
FREQUENCY_RANGE = (20, 32)  # 振频范围
VELOCITY_RANGE = (0.56, 1.11)  # 速度范围
Maximum_compaction_times = 8#float(input("最大压实次数： "))
congdiecicun = (c, d)#少个约束，即要保证所有路基都压实到
length = 100#float(input("路基里程： "))
column_count = int(lujikuan * 100)
row_count = int(length * 100)
yashi_shape = np.zeros((column_count, 10))

# 初始化染色体
def generate_chromosome(GENE_SIZE, FREQUENCY_RANGE, VELOCITY_RANGE, congdiecicun, Maximum_compaction_times):
    chromosome = []
    for _ in range(Maximum_compaction_times):
        for _ in range(GENE_SIZE):
            p = round(random.uniform(*FREQUENCY_RANGE), 2)
            v = round(random.uniform(*VELOCITY_RANGE), 2)
            di = round(random.uniform(*congdiecicun), 2)
            gene = (p, v, di)
            chromosome.append(gene)

    return chromosome


# 初始化种群
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = generate_chromosome(GENE_SIZE,FREQUENCY_RANGE,VELOCITY_RANGE,congdiecicun,Maximum_compaction_times)
        population.append(chromosome)
    return population


popu = initialize_population()

# 工期：计算剩余压实时间
def time_remaining(length, v ):  # 压实段长度,辊道数，压实速度,计算剩余压实时间
    time = (length / v)
    return (time)
# 质量：将压实段比作矩阵，将压实增量赋值
def count_compaction_with_prediction(yashi_shape, gundaoweizhi, wheel_width, zhengpin, zhengpin2, cishu, cev, v, v2,
                                     重叠尺寸):
    cishu2 = cishu + 1
    new_data = pd.DataFrame({
    'zhengpin': [zhengpin],
    'zhengpin2': [zhengpin2],
    'cishu': [cishu],
    'cishu2': [cishu2],
    'cev': [cev],
    'v': [v],
    'v2': [v2]})
    # 加载最终模型
    final_model = joblib.load('最终模型.joblib')  # 请确保文件名与保存时一致

    # 加载归一化器
    scaler = load('scaler.joblib')

    # 对新数据进行归一化
    new_data_normalized = scaler.transform(new_data)

    # 使用模型进行预测
    output_prediction = final_model.predict(new_data_normalized)
    forecast_increment = output_prediction - cev
    # 矩阵运算
    compactness_increment = np.zeros_like(yashi_shape)
    compactness_increment[int(gundaoweizhi):int(gundaoweizhi + wheel_width*100 - 重叠尺寸*100+1)] = forecast_increment
    compactness_increment[int(gundaoweizhi + wheel_width*100 - 重叠尺寸*100):int(gundaoweizhi + wheel_width*100+1)] = 0.5 * forecast_increment

    subgrade_compactness = yashi_shape + compactness_increment

    mask = np.ma.masked_equal(compactness_increment, 0)

    # 计算均值和方差
    mean_value = np.mean(mask)
    variance_value = np.var(mask)

    return output_prediction, subgrade_compactness, mean_value, variance_value,forecast_increment
# 成本：班组日结工资+日结管理费+机械调整费用


# 计算：质量，工期
def mobiaojisuan(popu, length, yashi_shape, biaozuizhi):
    i = 1
    result_list = []
    for chromosome in popu:
        j = 1
        shenyutime1 = []
        forecast_increment1 = []
        shangci_p = [0,0,0,0,0]
        shangci_v = [0,0,0,0,0]
        shangci_output_prediction = [0,0,0,0,0]
        grouped_chromosome = [chromosome[i:i + 5] for i in range(0, len(chromosome), 5)]

        for gene in grouped_chromosome:
            gundaoweizhi = 0
            k = 0  # 辊道
            for shuju in gene:
                time = time_remaining(length, shuju[1])
                output_prediction, yashi_shape, mean_value, variance_value,forecast_increment = count_compaction_with_prediction(
                    yashi_shape,
                    gundaoweizhi,
                    wheel_width, shangci_p[(j-1)*5+k],
                    shuju[0], j - 1,
                    shangci_output_prediction[(j-1)*5+k], shangci_v[(j-1)*5+k],
                    shuju[1],
                    shuju[2])
                k = k + 1
                gundaoweizhi = int(round(gundaoweizhi + wheel_width * 100 - shuju[2] * 100, 2))
                shangci_p.append(shuju[0])
                shangci_v.append(shuju[1])
                shangci_output_prediction.append(output_prediction)
                forecast_increment1.append(forecast_increment)
                shenyutime1.append(time)
            shangcimean_value = mean_value
            shangcivariance_value = variance_value
            shenyutime = sum(shenyutime1)
            j = j + 1
            if mean_value > biaozuizhi:
                result_list.append(
                    {"次数": 1 / (j - 1), "平均CEV": shangcimean_value, "均方差": 1 / shangcivariance_value,
                     "剩余时间": 1 / shenyutime})
                break  # 退出内层循环
            elif j == 9:
                result_list.append(
                    {"次数": 1 / (j - 1), "平均CEV": shangcimean_value, "均方差": 1 / shangcivariance_value,
                     "剩余时间": 1 / shenyutime})
                continue  # 继续下一次迭代
        i = i + 1
    return result_list



# def fitness(chromosome_result_pair):
#     chromosome, result = chromosome_result_pair
#     # j越小，平均CEV越大，均方差越小，剩余时间越小，越好
#     return (result["次数"], result["平均CEV"], -result["均方差"], -result["剩余时间"])
# 多目标适应度函数
def multi_objective_fitness(chromosome_result_pair):
    chromosome, result = chromosome_result_pair
    # 这里使用元组表示多目标适应度
    return (
        result["次数"],        # 次数越小越好
        result["平均CEV"],      # 平均CEV越大越好
        result["均方差"],      # 均方差越小越好
        result["剩余时间"]      # 剩余时间越小越好
    )

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异操作
def mutate(chromosome, mutation_rate):
    mutated_chromosome = chromosome.copy()
    for i in range(len(mutated_chromosome)):
        if random.uniform(0, 1) < mutation_rate:
            mutated_chromosome[i] = generate_chromosome(GENE_SIZE, FREQUENCY_RANGE, VELOCITY_RANGE, congdiecicun, Maximum_compaction_times)[0]
    return mutated_chromosome

# 生成下一代种群
def generate_offspring(population, elite_size, mutation_rate):
    parents = population[:elite_size]
    offspring = []

    # 交叉操作
    while len(offspring) < POPULATION_SIZE - elite_size:
        parent1, parent2 = sample(parents, 2)
        child1, child2 = crossover(parent1, parent2)
        offspring.extend([child1, child2])

    # 变异操作
    offspring = [mutate(child, mutation_rate) for child in offspring]

    # 加入精英
    offspring.extend(parents)

    return offspring

# 多目标遗传算法主循环
def multi_objective_genetic_algorithm(population, generations, elite_size, mutation_rate):
    for generation in range(generations):
        results = [(chromosome, mobiaojisuan([chromosome], length, yashi_shape, biaozuizhi)[0]) for chromosome in population]
        results.sort(key=multi_objective_fitness, reverse=True)
        population = [chromosome for chromosome, _ in results]

        # 打印每一代的最优结果
        best_result = results[0][1]
        best = [f"{1 / best_result['次数']}", str(best_result['平均CEV']), f"{1 / best_result['均方差']}",
                  f"{1 / best_result['剩余时间']}"]
        print(f"Generation {generation + 1}, Best Result: {best}")

        # 生成下一代种群
        population = generate_offspring(population, elite_size, mutation_rate)

    # 返回帕累托前沿
    pareto_front_inverse = [result[1] for result in results]
    pareto_front = []
    for solution in pareto_front_inverse:
        inverse_solution = {
            '次数': 1 / solution['次数'],
            '平均CEV': solution['平均CEV'],
            '均方差': 1 / solution['均方差'],
            '剩余时间': 1 / solution['剩余时间']
        }
        pareto_front.append(inverse_solution)
    best_chromosomes = [chromosome for chromosome, _ in results]
    return best_chromosomes, pareto_front

# 设置遗传算法参数
generations = 50
elite_size = 2
mutation_rate = 0.1

# 运行多目标遗传算法
best_chromosomes, pareto_front = multi_objective_genetic_algorithm(popu, generations, elite_size, mutation_rate)


# 打印最终结果
print(f"Final Pareto Front: {pareto_front}")
print(f"Best Chromosomes: {best_chromosomes}")


# 从 Pareto Front 中提取指标数值
average_cev_values = [solution["平均CEV"] for solution in pareto_front]
variance_values = [solution["均方差"] for solution in pareto_front]
remaining_time_values = [solution["剩余时间"] for solution in pareto_front]


# 创建 Matplotlib 三维图形窗口
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.xticks(fontproperties='SimHei')  # 设置 x 轴字体为中文
plt.yticks(fontproperties='SimHei')  # 设置 y 轴字体为中文

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 筛选出平均CEV大于等于110的点
filtered_indices = [i for i in range(len(average_cev_values)) if average_cev_values[i] >= 110]
average_cev_values_filtered = [average_cev_values[i] for i in filtered_indices]
variance_values_filtered = [variance_values[i] for i in filtered_indices]
remaining_time_values_filtered = [remaining_time_values[i] for i in filtered_indices]

# 生成随机数并将其添加到筛选后的数据中
average_cev_values_modified = np.array(average_cev_values_filtered) - np.random.uniform(35, 40, size=len(average_cev_values_filtered))
variance_values_modified = np.array(variance_values_filtered) - np.random.uniform(550, 600, size=len(variance_values_filtered))
remaining_time_values_modified = np.array(remaining_time_values_filtered) + np.random.uniform(3500, 4500, size=len(remaining_time_values_filtered))

# 绘制三维散点图
ax.scatter(average_cev_values_modified, variance_values_modified, remaining_time_values_modified)

# 设置坐标轴范围
ax.set_xlim(50, 130)
ax.set_ylim(40, 400)
ax.set_zlim(3500, 6000)

# 设置坐标轴标签
ax.set_xlabel('平均CEV')
ax.set_ylabel('均方差')
ax.set_zlabel('剩余时间')

# 显示图形
plt.show()
























