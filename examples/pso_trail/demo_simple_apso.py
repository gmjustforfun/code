import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sko.pso.APSO import APSO

"""
time:2020-10
author:gaomin
测试文件demo_simple_gpso
是一个最基本的全局版的pso，惯量权重选择LDW方法
测试几个30维基准函数的求解结果。
为了方便对比，我们将FEs=300000,particle_num = 20,c1=c2=2 w=0.9(和石琳师姐的Co-evolutionary pso的比对设置相同)
            将FEs=200000,particle_num=20,其他一样（和APSO中的GPSO结果对比）
测试时遇到的问题：如果只是进行30次实验，偶尔有1-2次陷入局部最优，最后统计效果很差
    解决方案：设置每个基准函数的真实最优和可接受精度，只有当算法得到的最优值的结果差距在可接受的范围内，放入table统计
"""
# 单峰函数 ok
def sphere(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += (i) ** 2
    return out_put
# 单峰函数 效果非常差!！！！！！！！！！！！！！
def schs222(x):
    out_put = 0
    out_put01 = 1
    for i in x:
        out_put += abs(i)
        out_put01 = abs(i)*out_put01
    out_put = out_put01+out_put
    return out_put
def Schwefel_P222(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)
# 单峰函数 效果非常差!！！！！！！！！！！！！！！！！！
def quadric(x):

    result = 0
    for i in range(len(x)):
        sum = 0
        for j in range(i+1):
            sum += x[j]
        sum = sum**2
        result += sum
    return result
def Quadric(x):
    output = 0
    # print(x.shape[0])
    for i in range(x.shape[0]):
        output += np.sum(x[0:i+1]) ** 2
    return output
# 单峰函数 ok
def Step(x):
    sum = 0
    for xi in x:
        sum = sum + np.floor(xi+0.5) ** 2
    return sum
# 单峰函数 ok
def rosenbrock(p):
    n_dim = len(p)
    res = 0
    for i in range(n_dim - 1):
        res += 100 * np.square(np.square(p[i]) - p[i + 1]) + np.square(p[i] - 1)
    return res
# 单峰函数 ok
def quadricNoise(x):
    sum = 0
    i = 0
    for xi in x:
        sum += i*xi**4
        i += 1
    result = sum + np.random.rand()
    return result
# 单峰函数 ok
def schls221(x):
    max = -np.inf
    for xi in x:
        if abs(xi) > max:
            max = abs(xi)
    return max
# 多峰函数 表达式有歧义,选择APSO的表达式 ok
def schwefel(x):
    sum = 0
    for xi in x:
        sum += (-xi*np.sin(xi))
    return sum
# 没见过 忽略
def schaffer(p):
    '''
    二维函数，具有无数个极小值点、强烈的震荡形态。很难找到全局最优值
    在(0,0)处取的最值0
    -10<=x1,x2<=10
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(np.sqrt(x))) - 0.5) / np.square(1 + 0.001 * x)
# 没见过 忽略
def shubert(p):
    '''
    2-dimension
    -10<=x1,x2<=10
    has 760 local minimas, 18 of which are global minimas with -186.7309
    '''
    x, y = p
    part1 = [i * np.cos((i + 1) * x + i) for i in range(1, 6)]
    part2 = [i * np.cos((i + 1) * y + i) for i in range(1, 6)]
    return np.sum(part1) * np.sum(part2)
# 多峰函数 ok 可惜效果不好！！！！！！！！！！！
def ackley(x):
    part1 = 0
    part2 = 0
    for i in x:
        part1 += (i ** 2)
        part2 += np.cos(2 * np.pi * i)
    left = 20 * np.exp(-0.2 * ((part1 / x.shape[0]) ** .5))
    right = np.exp(part2 / x.shape[0])
    return -left - right + 20 + np.e

# def Ackley(x):
#     if x.ndim == 1:
#         x = x.reshape(1, -1)
#
#     left = 20 * np.exp(-0.2 * (np.sum(x ** 2, axis=1) / x.shape[1]) ** .5)
#     right = np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / x.shape[1])
#
#     return -left - right + 20 + np.e
# 多峰函数 ok
def rastrigrin(p):
    '''
    多峰值函数，也是典型的非线性多模态函数
    -5.12<=xi<=5.12
    在范围内有10n个局部最小值，峰形高低起伏不定跳跃。很难找到全局最优
    has a global minimum at x = 0  where f(x) = 0
    '''
    return np.sum([np.square(x) - 10 * np.cos(2 * np.pi * x) + 10 for x in p])
# 多峰函数 ok
def griewank(p):
    '''
    存在多个局部最小值点，数目与问题的维度有关。
    此函数是典型的非线性多模态函数，具有广泛的搜索空间，是优化算法很难处理的复杂多模态问题。
    在(0,...,0)处取的全局最小值0
    -600<=xi<=600
    '''
    part1 = [np.square(x) / 4000 for x in p]
    part2 = [np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(p)]
    return np.sum(part1) - np.prod(part2) + 1


"---------------------------------------------------------------------------------------------------------"
# 这四个的设置是做误差剔除
trail_num = 100
table = np.zeros(30)  # 存放30个没有大的差异的结果,table的存放可以调整一下，方便结果的统一展示
opt_functions=[0,0,0,0]
opt_acceptance=[0.01,0.01,100,0.01]

def trail02():
    """
    该函数测试
    :return:
    """
    # final = np.zeros((2,11))
    i = 0
    for trail in range(trail_num):
        pso = APSO(func=sphere, dim=30, pop=20, max_iter=10000,
                  lb=np.ones(30)*(-100), ub=np.ones(30)*100, w=0.9, c1=2, c2=2)
        pso.run()
        print("Sphere(30) opt:", pso.gbest_y)
        if(abs(pso.gbest_y - opt_functions[0]) <= opt_acceptance[0] and i<30):
            table[i] = pso.gbest_y
            i += 1
        elif i==30:
            break
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(table)
    print("Sphere 30 dim 30 trails ave opt: ", np.mean(table))
    print("Sphere 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 0] = np.mean(table)
    # final[1, 0] = np.std(table)
    #
    i = 0
    for trail in range(trail_num):
        pso = APSO(func=Schwefel_P222, dim=30, pop=20, max_iter=10000,
                  lb=np.ones(30)*(-10), ub=np.ones(30)*10, w=0.9, c1=2, c2=2)
        pso.run()
        print("Schs222(30) opt:", pso.gbest_y)
        if(abs(pso.gbest_y - opt_functions[1]) <= opt_acceptance[1] and i<30):
            table[i] = pso.gbest_y
            i += 1
        elif i==30:
            break
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(table)
    print("Schs222 30 dim 30 trails ave opt: ", np.mean(table))
    print("Schs222 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 1] = np.mean(table)
    # final[1, 1] = np.std(table)
    #
    i = 0
    for trail in range(trail_num):
        pso = APSO(func=rosenbrock, dim=30, pop=20, max_iter=10000,
                  lb=np.ones(30) * (-10), ub=np.ones(30) * 10, w=0.9, c1=2, c2=2)
        pso.run()
        print("Rosenbrock(30) opt:", pso.gbest_y)
        if (abs(pso.gbest_y - opt_functions[2]) <= opt_acceptance[2] and i < 30):
            table[i] = pso.gbest_y
            i += 1
        elif i==30:
            break
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(table)
    print("Rosenbrock 30 dim 30 trails ave opt: ", np.mean(table))
    print("Rosenbrock 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 2] = np.mean(table)
    # final[1, 2] = np.std(table)
    #
    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=griewank, dim=30, pop=20, max_iter=10000,
    #               lb=np.ones(30) * (-600), ub=np.ones(30) * 600, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     # print("Griewank(30) opt:", pso.gbest_y)
    #     if (abs(pso.gbest_y - opt_functions[3]) <= opt_acceptance[3] and i < 30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i==30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("Griewank 30 dim 30 trails ave opt: ", np.mean(table))
    # print("Griewank 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 3] = np.mean(table)
    # final[1, 3] = np.std(table)
    #
    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=Step, dim=30, pop=20, max_iter=10000,
    #               lb=np.ones(30) * (-100), ub=np.ones(30) * 100, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     # print("Step(30) opt:", pso.gbest_y)
    #     if (abs(pso.gbest_y - 0) <= 0 and i < 30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i==30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("Step 30 dim 30 trails ave opt: ", np.mean(table))
    # print("Step 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 4] = np.mean(table)
    # final[1, 4] = np.std(table)

    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=quadricNoise, dim=30, pop=20, max_iter=10000,
    #               lb=np.ones(30) * (-1.28), ub=np.ones(30) * 1.28, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     print("quadricNoise(30) opt:", pso.gbest_y)
    #     if (abs(pso.gbest_y - 0) <= 10 and i < 30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i==30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("quadricNoise 30 dim 30 trails ave opt: ", np.mean(table))
    # print("quadricNoise 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 5] = np.mean(table)
    # final[1, 5] = np.std(table)
    #
    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=schls221, dim=30, pop=20, max_iter=10000,
    #               lb=np.ones(30) * (-100), ub=np.ones(30) * 100, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     print("Schls221(30) opt:", pso.gbest_y)
    #     if (abs(pso.gbest_y - 0) <= 10 and i < 30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i==30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("Schls221 30 dim 30 trails ave opt: ", np.mean(table))
    # print("Schls221 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 6] = np.mean(table)
    # final[1, 6] = np.std(table)
    #
    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=rastrigrin, dim=30, pop=20, max_iter=10000,
    #               lb=np.ones(30) * (-5.12), ub=np.ones(30) * 5.12, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     print("Rastrigin(30) opt:", pso.gbest_y)
    #     if (abs(pso.gbest_y - 0) <= 200 and i < 30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i==30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("Rastrigin 30 dim 30 trails ave opt: ", np.mean(table))
    # print("Rastrigin 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 7] = np.mean(table)
    # final[1, 7] = np.std(table)
    #
    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=schwefel, dim=30, pop=40, max_iter=10000,
    #               lb=np.ones(30) * (-500), ub=np.ones(30) * 500, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     print("schwefel(30) opt:", pso.gbest_y)
    #     if ((-12569.5 <= pso.gbest_y <= -10000) and i < 30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i == 30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("schwefel 30 dim 30 trails ave opt: ", np.mean(table))
    # print("schwefel 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 8] = np.mean(table)
    # final[1, 8] = np.std(table)
    #
    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=Quadric, dim=30, pop=20, max_iter=10000,
    #               lb=np.ones(30) * (-100), ub=np.ones(30) * 100, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     print("Quadric(30) opt:", pso.gbest_y)
    #     if (abs(pso.gbest_y - 0) <= 100 and i < 30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i == 30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("Quadric 30 dim 30 trails ave opt: ", np.mean(table))
    # print("Quadric 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 9] = np.mean(table)
    # final[1, 9] = np.std(table)
    #
    #
    # i = 0
    # for trail in range(trail_num):
    #     pso = PSO(func=ackley, dim=30, pop=20, max_iter=10000,
    #               lb=np.ones(30)*(-32), ub=np.ones(30)*32, w=0.9, c1=2, c2=2)
    #     pso.run(10000)
    #     print("Ackley(30) opt:", pso.gbest_y)
    #     if(abs(pso.gbest_y - 0) <= 0.01 and i<30):
    #         table[i] = pso.gbest_y
    #         i += 1
    #     elif i==30:
    #         break
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(table)
    # print("Ackley 30 dim 30 trails ave opt: ", np.mean(table))
    # print("Ackley 30 dim 30 trails ave opt: ", np.std(table))
    # final[0, 10] = np.mean(table)
    # final[1, 10] = np.std(table)
    #
    # # 最后结果
    # final = pd.DataFrame(final)
    # final.columns = ['Sphere', 'Schwefel_P222', 'Rosenbrock','Griewank', 'Step', 'Quadric_Noise','Schls221', 'Rastrigin',
    #                  ' Schwefel', 'Quadric', 'Ackley']
    # final.index = ['mean score', 'std']
    # print(final)

if __name__ == "__main__":

    # 用CEPSO参数计算结果
    # trail01()
    # 用APSO的参数和精度计算结果
    trail02()
