from pso.GPSO import GPSO
import numpy as np
import time
import pandas as pd

np.random.seed(42)


# f1 完成
def Sphere(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put
# f2 完成
def Sch222(x):
    out_put = 0
    out_put01 = 1
    for i in x:
        out_put += abs(i)
        out_put01 = abs(i)*out_put01
    out_put = out_put01+out_put
    return out_put

# f3 完成
def Quadric(x):
    output = 0
    # print(x.shape[0])
    for i in range(x.shape[0]):
        output += (np.sum(x[0:i+1])) ** 2
        # print(np.square(np.sum(x[0:i+1])))
    return output

# f4 完成
def Schl(x):
    # print(np.max(np.abs(x)))
    return np.max(np.abs(x))

# f5 完成
def Step(x):
    output = 0
    for i in x:
        output += (np.floor(i+0.5))**2
    return output

# f6 完成
def Noise(x):
    output = 0
    cnt = 1
    for i in x:
        output = cnt * (i**4) + output
        cnt += 1
    output += np.random.rand()
    return output
# f7 完成
def Rosenbrock(p):
    '''
    -2.048<=xi<=2.048
    函数全局最优点在一个平滑、狭长的抛物线山谷内，使算法很难辨别搜索方向，查找最优也变得十分困难
    在(1,...,1)处可以找到极小值0
    :param p:
    :return:
    '''
    n_dim = len(p)
    res = 0
    for i in range(n_dim - 1):
        res += 100 * np.square(np.square(p[i]) - p[i + 1]) + np.square(p[i] - 1)
    return res

# f8 有问题，忽略,这个是APSO的f8
def Schewel(x):
    out_put = 0
    for i in x:
        out_put += -i*np.sin(np.sqrt(abs(i)))
    return out_put

# f9 完成
def Rastrigin(p):
    '''
    多峰值函数，也是典型的非线性多模态函数
    -5.12<=xi<=5.12
    在范围内有10n个局部最小值，峰形高低起伏不定跳跃。很难找到全局最优
    has a global minimum at x = 0  where f(x) = 0
    '''
    return np.sum([np.square(x) - 10 * np.cos(2 * np.pi * x) + 10 for x in p])

# f10
def Ackley(x):
    part1 = 0
    part2 = 0
    for i in x:
        part1 += (i**2)
        part2 += np.cos(2 * np.pi * i)
    left = 20 * np.exp(-0.2 * ((part1 / x.shape[0]) ** .5))
    right = np.exp(part2 / x.shape[0])
    return  -left - right + 20 + np.e


# f11  ok
def Griewank(p):
    '''
    存在多个局部最小值点，数目与问题的维度有关。
    此函数是典型的非线性多模态函数，具有广泛的搜索空间，是优化算法很难处理的复杂多模态问题。
    在(0,...,0)处取的全局最小值0
    -600<=xi<=600
    '''
    part1 = [np.square(x) / 4000 for x in p]
    part2 = [np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(p)]
    return np.sum(part1) - np.prod(part2) + 1


g = 10000
times = 30
table = np.zeros((2, 10))
gBest = np.zeros((10, 30))  # 1010个函数的30次的最优值
for i in range(times):
    optimizer = GPSO(func=Sphere, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-100), ub=np.ones(30) * 100,
                     w=0.9, c1=2, c2=2, acceptance=0.01)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Sphere:', optimizer.gbest_y)
    table[0, 0] += optimizer.gbest_y
    table[1, 0] += end - start
    gBest[0, i] = optimizer.gbest_y

    optimizer = GPSO(func=Sch222, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-10), ub=np.ones(30) * 10,
                     w=0.9, c1=2, c2=2, acceptance=0.01)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Sch222:', optimizer.gbest_y)
    table[0, 1] += optimizer.gbest_y
    table[1, 1] += end - start
    gBest[1, i] = optimizer.gbest_y

    optimizer = GPSO(func=Quadric, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-100), ub=np.ones(30) * 100,
                     w=0.9, c1=2, c2=2, acceptance=100)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Quadric:', optimizer.gbest_y)
    table[0, 2] += optimizer.gbest_y
    table[1, 2] += end - start
    gBest[2, i] = optimizer.gbest_y

    optimizer = GPSO(func=Rosenbrock, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-10), ub=np.ones(30) * 10,
                     w=0.9, c1=2, c2=2, acceptance=100)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Rosenbrock:', optimizer.gbest_y)
    table[0, 3] += optimizer.gbest_y
    table[1, 3] += end - start
    gBest[3, i] = optimizer.gbest_y

    optimizer = GPSO(func=Step, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-100), ub=np.ones(30) * 100,
                     w=0.9, c1=2, c2=2, acceptance=0)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Step:', optimizer.gbest_y)
    table[0, 4] += optimizer.gbest_y
    table[1, 4] += end - start
    gBest[4, i] = optimizer.gbest_y

    optimizer = GPSO(func=Noise, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-1.28), ub=np.ones(30) * 1.28,
                     w=0.9, c1=2, c2=2, acceptance=0.01)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Noise:', optimizer.gbest_y)
    table[0, 5] += optimizer.gbest_y
    table[1, 5] += end - start
    gBest[5, i] = optimizer.gbest_y

    optimizer = GPSO(func=Schewel, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-500), ub=np.ones(30) * 500,
                     w=0.9, c1=2, c2=2, acceptance=-10000)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Schewel:', optimizer.gbest_y)
    table[0, 6] += optimizer.gbest_y
    table[1, 6] += end - start
    gBest[6, i] = optimizer.gbest_y

    optimizer = GPSO(func=Rastrigin, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-5.12), ub=np.ones(30) * 5.12,
                     w=0.9, c1=2, c2=2, acceptance=50)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Rastrigin:', optimizer.gbest_y)
    table[0, 7] += optimizer.gbest_y
    table[1, 7] += end - start
    gBest[7, i] = optimizer.gbest_y

    optimizer = GPSO(func=Ackley, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-32), ub=np.ones(30) * 32,
                     w=0.9, c1=2, c2=2, acceptance=0.01)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Ackley:', optimizer.gbest_y)
    table[0, 8] += optimizer.gbest_y
    table[1, 8] += end - start
    gBest[8, i] = optimizer.gbest_y

    optimizer = GPSO(func=Griewank, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-600), ub=np.ones(30) * 600,
                     w=0.9, c1=2, c2=2, acceptance=0.01)
    start = time.time()
    optimizer.run()
    end = time.time()
    print('Griewank:', optimizer.gbest_y)
    table[0, 9] += optimizer.gbest_y
    table[1, 9] += end - start
    gBest[9, i] = optimizer.gbest_y

table = table / times
table = pd.DataFrame(table)
table.columns = ['Sphere', 'Schwefel_P222', 'Quadric', 'Rosenbrock', 'Step', 'Quadric_Noise', 'Schwefel',
                 'Rastrigin', 'Ackley', 'Griewank']
table.index = ['mean score', 'mean time']
print(table)
print('10个测试函数的30次std:', np.std(gBest, axis=1))
print('10个测试函数的30次best:', np.min(gBest, axis=1))