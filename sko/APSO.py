#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from sko.tools import func_transformer
from .base import SkoBase
import random


def sphere_tv(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += (i-5) ** 2
    return out_put


def schwefels_tv(x):
    out_put = 0
    out_put01 = 1
    print(x)
    for i in x:
        out_put += abs(i-5)
        out_put01 = abs(i-5)*out_put01
    out_put = out_put01+out_put
    return out_put


def rosenbrock_tv(p):
    n_dim = len(p)
    res = 0
    for i in range(n_dim - 1):
        res += 100 * np.square(np.square(p[i]-5) - p[i + 1]-5) + np.square(p[i]-5 - 1)
    return res


def schewel_tv(x):
    out_put = 0
    for i in x:
        out_put += -(i-5)*np.sin(np.sqrt(abs(i-5)))
    return out_put


def sphere2dim1(x):
    '''
    this function is the target funvtion if "population_Distribution_Information_Of_PSO"
    r初始为5，在iter==50时，
    :param x: variable
    :param r: paremeters,the center of the sphere
    :return: result
    '''
    x1, x2 = x
    return (x1-5)**2+(x2-5)**2


class APSO(SkoBase):
    """
    Do APSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration

    w：the inertia weight
    d： array_like, shape is (pop,1)
        the mean distance of each particle i to all the other particles.

    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, func, dim=30, pop=100, max_iter=100, lb=None, ub=None, w=0.9, c1=2, c2=2, acceptance=0.01):
        """
        contruct function
        :param func: 目标函数
        :param dim:  维度
        :param pop:  粒子数目
        :param max_iter:  最大迭代次数
        :param lb: 上界
        :param ub: 下界
        :param w: 惯量权重
        :param c1: 加速系数1
        :param c2: 加速系数2
        :param acceptance:可接受精度
        """
        self.func = func_transformer(func)
        self.w = w                      # inertia
        self.cp, self.cg = c1, c2       # parameters to control personal best, global best respectively
        self.pop = pop                  # number of particles
        self.dim = dim                  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter        # max iter
        self.iter_num = 0               # current iter no
        self.acceptance = acceptance
        self.acceptance_iter = 0
        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        # 初始化粒子的位置
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        # 初始化粒子的速度
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y = self.cal_y()           # y = f(x) for all particles
        self.pbest_x = self.X.copy()    # personal best location of every particle in history
        self.pbest_y = self.Y.copy()    # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf           # global best y for all particles
        self.gbest_y_hist = []          # gbest_y of every iteration
        # 评估当前全体最优粒子
        self.update_gbest()
        # 实验2所需
        self.d = self.cal_d()  # the mean distance of each particle i to all the other particles.
        self.f = self.cal_f()  # 计算
        # record verbose values
        self.record_mode = True
        self.record_value = {'X': [], 'V': [], 'Y': [], 'gbest_each_generation': [], 'f': []}

    def update_V(self):
        """
        update V
        :return:
        """
        # 随机加速系数
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        # 加速公式
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        """
        update X
        :return:
        """
        # 更新各粒子的位置
        self.X = self.X + self.V
        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_d(self):
        """
        计算每个粒子到其他所有粒子的平均距离
         the mean distance of each particle i to all the other particles.
        :return:
        :return:
        """
        d = np.zeros(self.pop)
        for i in range(self.pop):  # 对每一个粒子
            sum = 0
            for j in range(self.pop):
                if j != i:
                    sum += np.sqrt(np.sum(np.square(self.X[i] - self.X[j])))
            d[i] = sum / (self.pop - 1)
        self.d = d
        return self.d

    def cal_y(self):
        """
        calculate Y and update Y
        :return:
        """
        # calculate y for every x in X
        # 计算每个粒子的适应值
        self.Y = self.func(self.X).reshape(-1, 1)
        # 每次更新了粒子位置后就可以计算欧式距离d和进化因子f
        self.cal_d()
        self.cal_f()
        return self.Y

    def cal_f(self):
        """
        计算每一代的进化因子f
        This function is used to calculate the value of evolutionary factor:f
        f = (dg - dmin) / (dmax - dmin)
        where dg :is the d of gbest particle
                dmin: the min of all d
                dmax:the min if all d
        every generation have one f
        :return:
        """
        # 找到全局最优的粒子的下标，锁定其对应的d
        dg = self.d[self.Y.argmin()]
        if (self.d.max() - self.d.min()) == 0:
            self.f = 0
        else:
            self.f = (dg - self.d.min()) / (self.d.max() - self.d.min())
        return self.f

    def update_pbest(self):
        '''
        update personal best，include pbest_x and pbest_y
        :return:
        '''
        # 更新每个粒子的个体最优值的位置和相应的适应值，越小越优
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        """
        global best
        :return:
        """
        # 更新全局最优的位置和适应值
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        '''
        record data of every generation
        :return:
        '''
        if not self.record_mode:
            return
        # 每一代记录一个X，shape(particles_num,dim)
        self.record_value['X'].append(self.X)
        # 每一代记录一个V，shape(particles_num,dim)
        self.record_value['V'].append(self.V)
        # 每一代记录一个Y，shape(particles_num,dim)
        self.record_value['Y'].append(self.Y)
        # 记录每一代的全局最优的位置 gbest_X,(particles_num,dim)
        self.record_value['gbest_each_generation'].append(self.X[self.Y.argmin(),:])
        # 计算每一代计算的进化因子f
        self.record_value['f'].append(self.f)

    def map(self,target_data):
        """

        :param target_data: 待归一化的数据
        :return:
        """

        MIN = 1.5
        MAX = 2.5
        d_min = 1  # 当前数据最大值
        d_max = 3  # 当前数据最小值
        return MIN + (MAX - MIN) / (d_max - d_min) * (target_data - d_min)

    def adaptiveParam(self):
        """
        This function is used to self-adaptive Param.
        Before performance this function,we have caculate evolutionary fctor f for all particles.
        use membership function of all 4 states,estimate state.
        According to the state ,adjust c,
        then adjust w.
        :return:
        """
        # step1 根据f估计状态state,对每一个f（一代只有一个）
        # 都能计算出四个状态值，比较四个状态值那个大（范围均为【0,1】），则此时的进化状态为哪一个。
        state = np.zeros(4)
        # S1
        if 0 <= self.f <= 0.4:
            state[0] = 0
        elif 0.4 < self.f <= 0.6:
            state[0] = 5 * self.f - 2
        elif 0.6 < self.f <= 0.7:
            state[0] = 1
        elif 0.7 < self.f <= 0.8:
            state[0] = -10 * self.f + 8
        else:
            state[0] = 0
        # S2
        if 0 <= self.f <= 0.2:
            state[1] = 0
        elif 0.2 < self.f <= 0.3:
            state[1] = 10 * self.f - 2
        elif 0.3 < self.f <= 0.4:
            state[1] = 1
        elif 0.4 < self.f <= 0.6:
            state[1] = -5 * self.f + 3
        else:
            state[1] = 0
        # S3
        if 0 <= self.f <= 0.1:
            state[2] = 1
        elif 0.1 < self.f <= 0.3:
            state[2] = -5 * self.f + 1.5
        else:
            state[2] = 0
        # S4
        if 0 <= self.f <= 0.7:
            state[3] = 0
        elif 0.7 < self.f <= 0.9:
            state[3] = 5 * self.f - 3.5
        else:
            state[3] = 1

        # 统计了四种状态值
        state_target = state.argmax()
        if state_target == 0:
            # print("这一代的状态为S1(explopration)")
            # tomorrow：加速系数的调整，检查，控制和限定
            # 随机生成参数【0.05,0.1】
            param = random.uniform(0.05, 0.1)
            # print(param)
            # 规范化  注意这里首先限制加速系数均为1.5-2.5区间-----------------------------------------------------------------
            cp = self.cp + param
            cg = self.cg + param
            cp = self.map(cp)
            cg = self.map(cg)
            if 3 <= cg+cp <= 4:
                self.cp = cp
                self.cg = cg
            else:
                cp = cp / (cp + cg) * 4
                cg = cg / (cp + cg) * 4
                self.cp = cp
                self.cg = cg
            # print("选择ELS for gBest")
            sigma = 1 - 0.9 * self.iter_num / self.max_iter
            P = self.X.copy()  # copy 此时的种群位置

            bestindex = self.Y.argmin()  # 锁定最优值位置
            gBest = self.X[bestindex][:]  # 单个粒子

            # 0-(dim-1) 维度随机数
            d = random.randint(0, self.dim-1)
            while(True):
                gBest[d] = gBest[d] + (self.ub[d] - self.lb[d]) * random.gauss(0, sigma)
                # 检查这一维是否满足上下界的要求
                if self.lb[d] <= gBest[d] <= self.ub[d]:  # 如果满足，计算新的适应值
                    P[bestindex][:] = gBest  # TypeError: 'builtin_function_or_method' object is not subscriptable
                    newfit = self.func(P)
                    if newfit[bestindex] < self.gbest_y:
                        self.X = P
                        self.Y = newfit
                        self.update_pbest()
                        self.update_gbest()
                    else:
                        self.X[self.Y.argmax()] = gBest
                        self.Y = self.cal_y()
                        self.update_pbest()
                        self.update_gbest()
                    break
                else:  # 如果不满足界的要求，重新找d,重新计算新的P
                    P = self.X.copy()
                    d = random.randint(0, self.dim-1)
        elif state_target == 1 or state_target == 2:
            # print("这一代的状态为S2(exploitation)或S3(convergence)")
            # 随机生成参数【0.05,0.5】
            param = random.uniform(0.05, 0.5)
            # print(param)
            # 规范化
            # print("增加c1,减少c2")
            # 规范化  注意这里首先限制加速系数均为1.5-2.5区间-----------------------------------------------------------------
            cp = self.cp + param
            cg = self.cg - param
            cp = self.map(cp)
            cg = self.map(cg)
            if 3 <= cg + cp <= 4:
                self.cp = cp
                self.cg = cg
            else:
                cp = cp / (cp + cg) * 4
                cg = cg / (cp + cg) * 4
                self.cp = cp
                self.cg = cg
        else:
            # 随机生成参数【0.05,0.1】
            # print("这一代的状态为S4(jumping outs)")
            param = random.uniform(0.05, 0.1)
            # print(param)
            # 规范化
            # print("减少c1,增加c2")
            # 规范化  注意这里首先限制加速系数均为1.5-2.5区间-----------------------------------------------------------------
            cp = self.cp - param
            cg = self.cg + param
            cp = self.map(cp)
            cg = self.map(cg)
            if 3 <= cg + cp <= 4:
                self.cp = cp
                self.cg = cg
            else:
                cp = cp / (cp + cg) * 4
                cg = cg / (cp + cg) * 4
                self.cp = cp
                self.cg = cg
        # adaptive w
        self.w = 1 / (1 + 1.5 * (np.exp(1) ** (-2.6 * self.f)))

    def run(self, max_iter=None):
        """
        main part
        :param max_iter:
        :return:
        """
        flag = False
        self.max_iter = max_iter or self.max_iter
        arange = (i for i in range(self.max_iter))
        for iter_num in arange:
            self.iter_num = iter_num
            print('iter time : %d' % (self.iter_num))
            # if部分是为了做实验2，如果不需要可以注释
            # if iter_num == 49:
            #     # 这四个语句是为了测试实验二，四个时变函数的进化因子变化图，
            #     # 如果要做实验2，根据目标函数选择即可
            #     self.func = func_transformer(sphere_tv)  # time-varying,改变目标函数
            #     # self.func = func_transformer(schwefels_tv)  # time-varying,改变目标函数
            #     # self.func = func_transformer(rosenbrock_tv)  # time-varying,改变目标函数
            #     # self.func = func_transformer(schewel_tv)  # time-varying,改变目标函数
            #
            #     self.cal_y()                      # 因为目标函数改变了，要重新计算此时种群对应的适应值
            #     self.pbest_y = self.Y.copy()      # 更新个体最优适应值
            #     self.pbest_x = self.X.copy()      # 更新个体最优位置
            #     self.update_pbest()               # 更新粒子的历史最优
            #     self.gbest_y = np.inf             # 此时新的历史最优设为inf
            self.adaptiveParam()  # 为下一代的更新自适应调整参数w,c,c2
            self.update_V()         # 更新速度
            self.recorder()         # 每一代记录一次X,Y,V
            self.update_X()         # 更新位置
            self.cal_y()            # 计算适应值
            self.update_pbest()     # 更新粒子的历史最优
            self.update_gbest()     # 更新全体历史最优
            if flag == False and self.gbest_y <= self.acceptance:
                flag = True
                self.acceptance_iter = iter_num
            self.gbest_y_hist.append(self.gbest_y)
            print('iter time : %d   w : %.8f%%  c1 : %.8f%%   c2 : %.8f%%  ' % (self.iter_num, self.w, self.cp, self.cg))
            print(self.gbest_y)

        return self



