import time

import matplotlib.pyplot as plt
import numpy as np
import random

from pso.CEPSO import CEPSO
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
        output += np.sum(x[0:i+1]) ** 2
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


class DemoTrailForCEPSO:
    def trailForSphere(self):
        """
        以f1函数实验一次，统计种群大小的变化趋势
        :return:
        """
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest  = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Sphere, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-100),
                              ub=np.ones(30)*100, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # # 种群变化图
        # plt.figure(0)
        # plt.plot(np.mean(pop01,axis=0))
        # plt.plot(np.mean(pop02, axis=0))
        # plt.show()
        # # 种群进化图
        # plt.figure(1)
        # P = np.mean(gbest_All, axis=0)
        # dev = np.std(gbest_All,axis=0)
        # plt.plot(P)
        # plt.show()
        print('f130次试验mean:',np.mean(gbest))     #
        print('f130次试验std:',np.std(gbest))
        print('f130次试验best:',np.min(gbest))

    def trailForF2(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Sch222, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-10),
                              ub=np.ones(30)*10, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f2 30次试验mean:',np.mean(gbest))     #
        print('f2 30次试验std:',np.std(gbest))
        print('f2 30次试验best:',np.min(gbest))

    def trailForF3(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Quadric, dim=30, pop01=20, pop02=20, max_iter=7500, lb=np.ones(30) * (-100),
                              ub=np.ones(30) * 100, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01, axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f3 30次试验mean:', np.mean(gbest))  #
        print('f3 30次试验std:', np.std(gbest))
        print('f3 30次试验best:', np.min(gbest))

    def trailForF4(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Schl, dim=30, pop01=20, pop02=20, max_iter=7500, lb=np.ones(30) * (-100),
                              ub=np.ones(30) * 100, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01, axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f4 30次试验mean:', np.mean(gbest))  #
        print('f4 30次试验std:', np.std(gbest))
        print('f4 30次试验best:', np.min(gbest))

    def trailForF5(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Step, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-100),
                              ub=np.ones(30)*100, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f5 30次试验mean:',np.mean(gbest))     #
        print('f5 30次试验std:',np.std(gbest))
        print('f5 30次试验best:',np.min(gbest))

    def trailForF6(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Noise, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-1.28),
                              ub=np.ones(30)*1.28, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f6 30次试验mean:',np.mean(gbest))     #
        print('f6 30次试验std:',np.std(gbest))
        print('f6 30次试验best:',np.min(gbest))

    def trailForF7(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Rosenbrock, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-10),
                              ub=np.ones(30)*10, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f7 30次试验mean:',np.mean(gbest))     #
        print('f7 30次试验std:',np.std(gbest))
        print('f7 30次试验best:',np.min(gbest))

    def trailForF8(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Schewel, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-10),
                              ub=np.ones(30)*10, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f8 30次试验mean:',np.mean(gbest))     #
        print('f8 30次试验std:',np.std(gbest))
        print('f8 30次试验best:',np.min(gbest))

    def trailForF9(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Rastrigin, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-5.12),
                              ub=np.ones(30)*5.12, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f9 30次试验mean:',np.mean(gbest))     #
        print('f9 30次试验std:',np.std(gbest))
        print('f9 30次试验best:',np.min(gbest))

    def trailForF10(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Ackley, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-32),
                              ub=np.ones(30)*32, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f10 30次试验mean:',np.mean(gbest))     #
        print('f10 30次试验std:',np.std(gbest))
        print('f10 30次试验best:',np.min(gbest))

    def trailForF11(self):
        pop01 = np.zeros((30, 7500))
        pop02 = np.zeros((30, 7500))
        gbest_GPSO = np.zeros((30, 7500))
        gbest_LPSO = np.zeros((30, 7500))
        gbest_All = np.zeros((30, 7500))
        gbest = np.zeros(30)
        for i in range(30):
            optimizer = CEPSO(func=Griewank, dim=30, pop01=20,pop02=20,max_iter=7500, lb=np.ones(30)*(-600),
                              ub=np.ones(30)*600, w=0.9, c1=2, c2=2)
            optimizer.run()
            pop01[i] = optimizer.record_value['pop01']
            pop02[i] = optimizer.record_value['pop02']
            gbest_GPSO[i] = optimizer.record_value['gbest_GPSO']
            gbest_LPSO[i] = optimizer.record_value['gbest_LPSO']
            gbest_All[i] = optimizer.record_value['gbest_All']
            gbest[i] = optimizer.gbestY
            print(optimizer.gbestY)

        # 种群变化图
        plt.figure(0)
        plt.plot(np.mean(pop01,axis=0))
        plt.plot(np.mean(pop02, axis=0))
        plt.show()
        # 种群进化图
        plt.figure(1)
        P = np.mean(gbest_All, axis=0)
        plt.plot(P)
        plt.show()
        print('f11 30次试验mean:',np.mean(gbest))     #
        print('f11 30次试验std:',np.std(gbest))
        print('f11 30次试验best:',np.min(gbest))

if __name__ == "__main__":
    # pso_TVIW()
    demo_01 = DemoTrailForCEPSO()
    # 实验1
    # demo_01.trailForSphere()  # ok
    # demo_01.trailForF2()
    demo_01.trailForF3()
    # demo_01.trailForF4()
    # demo_01.trailForF5()
    # demo_01.trailForF6()
    # demo_01.trailForF7()
    # demo_01.trailForF8()
    # demo_01.trailForF9()
    # demo_01.trailForF10()
    # demo_01.trailForF11()  # ok
    # demo_01.trailForF12()
    # demo_01.trailForF13()
    # x = np.array([1,2,3,4])
    # print(Quadric(x))
