import time

import matplotlib.pyplot as plt
import numpy as np
from pso.APSO import APSO
from pso.PSO import PSO
import pandas as pd
def sphere2dim(x):
    '''
    this function is the target funvtion if "population_Distribution_Information_Of_PSO"
    r初始为5，在iter==50时，
    :param x: variable
    :param r: paremeters,the center of the sphere
    :return: result
    '''
    x1, x2 = x
    return (x1+5)**2+(x2+5)**2


def sphere(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += (i+5) ** 2
    return out_put


def sphere01(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put


def schwefels(x):
    out_put = 0
    out_put01 = 1
    for i in x:
        out_put += abs(i+5)
        out_put01 = abs(i+5)*out_put01
    out_put = out_put01+out_put
    return out_put


def rosenbrock(p):
    n_dim = len(p)
    res = 0
    for i in range(n_dim - 1):
        res += 100 * np.square(np.square(p[i]+5) - p[i + 1]+5) + np.square(p[i]+5 - 1)
    return res


def schewel(x):
    out_put = 0
    for i in x:
        out_put += -(i+5)*np.sin(np.sqrt(abs(i+5)))
    return out_put

"----------------------------------"
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
        output += np.square(np.sum(x[0:i+1]))
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
# f8 有问题，忽略
# 这个是APSO的f8
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
    return -left - right + 20 + np.e
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



class DemoTrailForAPSO:
    def __init__(self, test_question=None):
        self.test_question = test_question  # 区分测试的内容是什么

    def population_Distribution_Information_Of_PSO(self):
        """
        实验1：
        This function is used to trail the population distribution information of PSO
        This function manipulate GPSO proposed in article "A modified PSO"
        The article does not give the specific parameters setting. We set w=0.8,c1=c2=2.
        MAXITER=100.
        The target function: f = (x1-r)^2+(x2-r)^2
        The constrict :xi [-10,10]
        This function plot three figures.
        “注意：文献中，当迭代到50代的时候，r由-5变为了5，
        在PSO主程序中加上判断语句，如果迭代次数满足条件了，self.func换掉
        现在计算结果是对的，但是从绘制出的种群分布图来看，到了50代的时候没能跳出去？？？？？？？
        已解决!!!
        :return:
        """
        "----------------------------1.计算，得到进化曲线图和结果"
        pso = PSO(func=sphere2dim, dim=2, pop=100, max_iter=100, lb=[-10, -10], ub=[10, 10], w=0.8, c1=1.5, c2=1)
        pso.run()
        # print(pso.record_value['gbest_each_generation'])  # 给出每一代的适应值，可以绘制进化曲线图
        print(pso.gbest_y_hist)
        plt.plot(pso.gbest_y_hist)
        plt.show()  # 绘制进化曲线图
        "------------------------------2.各代绘制种群分布图"
        plt.figure(1)
        ax1 = plt.subplot(2, 3, 1)
        plt.title('1代')
        ax2 = plt.subplot(2, 3, 2)
        plt.title('25代')
        ax3 = plt.subplot(2, 3, 3)
        plt.title('49代')
        ax4 = plt.subplot(2, 3, 4)
        plt.title('50代')
        ax5 = plt.subplot(2, 3, 5)
        plt.title('60代')
        ax6 = plt.subplot(2, 3, 6)
        plt.title('80代')
        # 选择ax1
        plt.sca(ax1)
        # matplotlib画图中中文显示会有问题，需要这两行设置默认字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 坐标轴，设置标签和范围
        plt.xlabel('x1')
        plt.xlabel('x2')
        plt.xlim(xmax=10, xmin=-10)
        plt.ylim(ymax=10, ymin=-10)
        # 设置颜色
        color = '#00CED1'
        # 设置点的面积
        area = np.pi * 2.5
        area1 = np.pi * 5
        "--------静态可以不要"
        # 第0代，初始分布
        for particle_no in range(pso.pop):
            plt.scatter(pso.record_value['X'][0][particle_no][0], pso.record_value['X'][0][particle_no][1], s=area, c=color, alpha=0.4)
        plt.scatter(pso.record_value['gbest_each_generation'][0][0],
                    pso.record_value['gbest_each_generation'][0][1], s=area1, c='#ff0000', alpha=0.4)
        # 选择ax2，第25代
        plt.sca(ax2)
        for particle_no in range(pso.pop):
            plt.scatter(pso.record_value['X'][24][particle_no][0], pso.record_value['X'][24][particle_no][1], s=area,
                        c=color, alpha=0.4)
        plt.scatter(pso.record_value['gbest_each_generation'][24][0],
                    pso.record_value['gbest_each_generation'][24][1], s=area1, c='#ff0000', alpha=0.4)
        # 选择ax3，第49代
        plt.sca(ax3)
        for particle_no in range(pso.pop):
            plt.scatter(pso.record_value['X'][48][particle_no][0], pso.record_value['X'][48][particle_no][1], s=area,
                        c=color, alpha=0.4)
        plt.scatter(pso.record_value['gbest_each_generation'][48][0],
                    pso.record_value['gbest_each_generation'][48][1], s=area1, c='#ff0000', alpha=0.4)
        # 选择ax4，第50代
        plt.sca(ax4)
        for particle_no in range(pso.pop):
            plt.scatter(pso.record_value['X'][50][particle_no][0], pso.record_value['X'][49][particle_no][1], s=area,
                        c=color, alpha=0.4)
        plt.scatter(pso.record_value['gbest_each_generation'][49][0],
                    pso.record_value['gbest_each_generation'][49][1], s=area1, c='#ff0000', alpha=0.4)
        # 选择ax5，第60代
        plt.sca(ax5)
        for particle_no in range(pso.pop):
            plt.scatter(pso.record_value['X'][59][particle_no][0], pso.record_value['X'][59][particle_no][1], s=area,
                        c=color, alpha=0.4)
        plt.scatter(pso.record_value['gbest_each_generation'][59][0],
                    pso.record_value['gbest_each_generation'][59][1], s=area1, c='#ff0000', alpha=0.4)
        # 选择ax6，第80代
        plt.sca(ax6)
        for particle_no in range(pso.pop):
            plt.scatter(pso.record_value['X'][79][particle_no][0], pso.record_value['X'][79][particle_no][1], s=area,
                        c=color, alpha=0.4)
        plt.scatter(pso.record_value['gbest_each_generation'][79][0],
                    pso.record_value['gbest_each_generation'][79][1], s=area1, c='#ff0000', alpha=0.4)
        plt.show()
        "--------绘制动态图，完成！"
        plt.ion()  # 开启交互模式
        plt.subplots()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.title(str(1))
        plt.scatter(pso.record_value['gbest_each_generation'][0][0],
                    pso.record_value['gbest_each_generation'][0][1], s=area1, c='#ff0000', alpha=0.4)
        for particle_no in range(pso.pop):
            plt.scatter(pso.record_value['X'][0][particle_no][0], pso.record_value['X'][0][particle_no][1], s=area, c=color, alpha=0.4)
        for generation_no in range(pso.max_iter-1):  # 对每一代0-98  2-100
            plt.clf()
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.title(str(generation_no+2))
            plt.scatter(pso.record_value['gbest_each_generation'][generation_no+1][0], pso.record_value['gbest_each_generation'][generation_no+1][1], s=area1, c='#ff0000', alpha=0.4)
            for particle_no in range(pso.pop):  # 绘制每个个体的散点分布
                plt.scatter(pso.record_value['X'][generation_no+1][particle_no][0], pso.record_value['X'][generation_no+1][particle_no][1],
                            s=area,
                            c=color, alpha=0.4)
            plt.pause(0.1)
        plt.ioff()
        plt.show()

    def evolutionary_information_f(self):
        """
        实验2
        This function is used to plot value evolution factor f
                                    no.     name        dim     range       optima    acceptance
        the menchbank function is   f1:     Sphere      30dim   [-100,100]     0         0.01                 time-varing
                                    f2:     Schwefel's  30dim   [-10,10]       0         0.01       单峰函数  time-varing
                                    f4:     Rosenbrock  30dim   [-10,10]       0         100        单峰函数  time-varing
                                    f7:     Schwefel    30dim   [-500,500]  -12569.5    -10000      多峰函数
        :return:
        要绘制四个函数的f变化图
        :return:
        """
        # f1函数2维时变的进化因子的变化趋势图
        pso = APSO(func=sphere2dim, dim=2, pop=100, max_iter=100, lb=np.ones(2)*(-10), ub=np.ones(2)*10, w=0.9, c1=2, c2=2, acceptance=0.01)
        pso.run()
        print(pso.record_value['f'])     # 给出每一代的适应值，可以绘制进化曲线图
        plt.plot(pso.record_value['f'])
        plt.show()                      # 绘制进化曲线图

        # f2函数2维时变的进化因子的变化趋势图
        # pso = APSO(func=schwefels, dim=2, pop=100, max_iter=100, lb=np.ones(2) * (-10), ub=np.ones(2) * 10, w=0.9,
        #            c1=2, c2=2, acceptance=None)
        # pso.run()
        # print(pso.record_value['f'])    # 给出每一代的适应值，可以绘制进化曲线图
        # plt.plot(pso.record_value['f'])
        # plt.show()                      # 绘制进化曲线图

        # f4函数2维时变的进化因子的变化趋势图
        # pso = APSO(func=rosenbrock, dim=2, pop=100, max_iter=100, lb=np.ones(2) * (-10), ub=np.ones(2) * 10, w=0.9,
        #            c1=2, c2=2, acceptance=None)
        # pso.run()
        # print(pso.record_value['f'])    # 给出每一代的适应值，可以绘制进化曲线图
        # plt.plot(pso.record_value['f'])
        # plt.show()                      # 绘制进化曲线图

        # f7多封函数的进化因子变化曲线图
        # pso = APSO(func=schewel, dim=2, pop=100, max_iter=100, lb=np.ones(2) * (-500), ub=np.ones(2) * 500, w=0.9,
        #            c1=2, c2=2, acceptance=None)
        # pso.run()
        # print(pso.record_value['f'])  # 给出每一代的适应值，可以绘制进化曲线图
        # plt.plot(pso.record_value['f'])
        # plt.show()  # 绘制进化曲线图

    def lala(self):

        # vec1 = np.array([2,3,2])
        # vec2 = np.array([4,8,1])
        # d = np.sqrt(np.sum(np.square(vec2 - vec1)))
        # print(d)
        # d = np.sqrt(30)
        # print(d)

        # a = np.array([2,3,1,5,6])
        # print(a.min())
        # print(a.argmin())

        # zoom_o2_5 = Interval(2, 5) #闭区间
        #
        # print(zoom_o2_5)
        #
        # print(2 in zoom_o2_5)
        #
        # print(5 in zoom_o2_5)
        # param = random.uniform(0.05, 0.1)
        # print(param)


        X = np.random.uniform(low=-10, high=10, size=(10, 5))
        print(X)
        print(X[1][:])
        X[2][:]=X[1][:]
    def trail_APSO_mean_FEs(self):
        """
        统计GPSO的平均迭代次数，最大迭代次数为200000，试验30次，有精度要求

        :return:
        """
        # FE = []    # 记录30次试验的达到精度的迭代次数，需要统计平均，表3
        # BEST = []  # 记录30次试验的最好解 用于统计均值和标准差 表VI
        #
        # arange = (i for i in range(1))
        # time_start = time.process_time()
        # for x in arange:
        #     print("正在运行中----------------------")
        #     pso = APSO(func=sphere01, dim=30, pop=20, max_iter=10000, lb=np.ones(30) * (-100), ub=np.ones(30) * 100, w=0.9, c1=2, c2=2, acceptance=0.01)
        #     pso.run()
        #     FE.append(pso.acceptance_iter)
        #     BEST.append(pso.gbest_y)
        # time_end = time.process_time()
        # print('the time for 30 GPSO trials  of f1 is %d ' %(time_end-time_start))
        # print('MEAN FEs IN OBTAINING ACCEPTABLE SOLUTIONS BY GPSO WITHOUT PARAMETER ADAPTATION %d ' % (np.mean(FE)))
        # print('MEAN SOLUTIONS OF THE 30 TRIALS OBTAINED BY GPSO WITHOUT PARAMETER ADAPTATION F1:%d'%(np.mean(BEST)))
        # print('STD OF THE 30 TRIALS OBTAINED BY GPSO WITHOUT PARAMETER ADAPTATION F1:%d' % (np.std(BEST)))
        # print('Best SOLUTIONS OF THE 30 TRIALS OBTAINED BY GPSO WITHOUT PARAMETER ADAPTATION F1:%d' % (np.min(BEST)))
        # print(FE)
        # print(BEST)


        g = 10000
        times = 30
        table = np.zeros((2, 10))
        for i in range(times):
            optimizer = APSO(func=sphere01, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-100), ub=np.ones(30) * 100,
                       w=0.9, c1=2, c2=2, acceptance=0.01)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 0] += optimizer.gbest_y
            table[1, 0] += end - start

            optimizer = APSO(func=Sch222, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-10), ub=np.ones(30) * 10,
                             w=0.9, c1=2, c2=2, acceptance=0.01)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 1] += optimizer.gbest_y
            table[1, 1] += end - start

            optimizer = APSO(func=Quadric, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-100), ub=np.ones(30) * 100,
                             w=0.9, c1=2, c2=2, acceptance=100)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 2] += optimizer.gbest_y
            table[1, 2] += end - start

            optimizer = APSO(func=Rosenbrock, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-10), ub=np.ones(30) * 10,
                             w=0.9, c1=2, c2=2, acceptance=100)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 3] += optimizer.gbest_y
            table[1, 3] += end - start

            optimizer = APSO(func=Step, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-100), ub=np.ones(30) * 100,
                             w=0.9, c1=2, c2=2, acceptance=0)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 4] += optimizer.gbest_y
            table[1, 4] += end - start

            optimizer = APSO(func=Noise, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-1.28), ub=np.ones(30) * 1.28,
                             w=0.9, c1=2, c2=2, acceptance=0.01)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 5] += optimizer.gbest_y
            table[1, 5] += end - start

            optimizer = APSO(func=Schewel, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-500), ub=np.ones(30) * 500,
                             w=0.9, c1=2, c2=2, acceptance=-10000)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 6] += optimizer.gbest_y
            table[1, 6] += end - start


            optimizer = APSO(func=Rastrigin, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-5.12), ub=np.ones(30) * 5.12,
                             w=0.9, c1=2, c2=2, acceptance=50)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 7] += optimizer.gbest_y
            table[1, 7] += end - start

            # x_max = 5.12 * np.ones(d)
            # x_min = -5.12 * np.ones(d)
            # optimizer = APSO(fit_func=Noncontinuous_Rastrigin, num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
            # start = time.time()
            # optimizer.opt()
            # end = time.time()
            # table[0, 8] += optimizer.gbest_y
            # table[1, 8] += end - start
            #
            #
            optimizer = APSO(func=Ackley, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-32), ub=np.ones(30) * 32,
                             w=0.9, c1=2, c2=2, acceptance=0.01)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 8] += optimizer.gbest_y
            table[1, 8] += end - start


            optimizer = APSO(func=Griewank, dim=30, pop=20, max_iter=g, lb=np.ones(30) * (-600), ub=np.ones(30) * 600,
                             w=0.9, c1=2, c2=2, acceptance=0.01)
            start = time.time()
            optimizer.run()
            end = time.time()
            table[0, 9] += optimizer.gbest_y
            table[1, 9] += end - start

        table = table / times
        print(table)
        table = pd.DataFrame(table)
        table.columns = ['Sphere', 'Schwefel_P222', 'Quadric', 'Rosenbrock', 'Step', 'Quadric_Noise', 'Schwefel',
                         'Rastrigin', 'Ackley', 'Griewank']
        table.index = ['mean score', 'mean time']
        print(table)


if __name__ == "__main__":
    # pso_TVIW()
    demo_01 = DemoTrailForAPSO()
    # 实验1 GPSO种群分布实验
    # demo_01.population_Distribution_Information_Of_PSO()
    # test detail
    # demo_01.lala()

    # 实验2:GPSO的进化因子变化实验
    # 注意：做实验2要将APSO代码处的自适应部分注释
    # demo_01.evolutionary_information_f()

    # 实验3
    demo_01.trail_APSO_mean_FEs()




