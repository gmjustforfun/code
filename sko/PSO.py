#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from sko.tools import func_transformer
from .base import SkoBase
from .operators import pso_inertia




class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.

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


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self,func,dim=10,pop=40,max_iter=1000,lb=None, ub=None,w=0.8,c1=2,c2=2):
        self.func = func_transformer(func) # 格式化后的测试函数即适应值函数

        self.w = w          # inertia 惯性因子
        self.w_max=0.9
        self.w_min=0.4

        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles 粒子的数目
        self.dim = dim  # dimension of particles, which is the number of variables of func 函数的维度
        self.max_iter = max_iter  # max iteration
        self.iter_num = 1  # current iteration

        self.has_constraints = not (lb is None and ub is None) # 布尔变量，如果有上下界约束:true
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        # 如果if满足做前面的[-1. -1. -1.]，否则做else[ 0.  -1.   0.5],即用本身构造数组，ub同理
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        # assert的意思是，后面表达式应该是True，否则，后面的代码肯定会出错。
        # 如果断言失败，assert语句本身就会抛出AssertionError：
        # 启动Python解释器时可以用 - O参数来关闭asser

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))#随机生成pop*dim的X变量且满足上下界要求

        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))# 随机生成粒子的速度，pop*dim ,每个粒子的速度向量都有dim维。

        self.Y = self.cal_y()
        #计算初始适应值 y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        # 将初始值作为个体粒子的最优值
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        # 记录粒子的最优y（适应值）
        self.gbest_x = np.zeros((1, self.dim))
        # 全局最优位置，初始为[[0. 0. 0.]]:global best location for all particles
        self.gbest_y = np.inf
        # 全局最优的适应值y:global best y for all particles
        self.gbest_y_hist = []
        # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def cal_w(self,type):
        '''
        根据不同的方法调整w
        Sevaral stratages to adjust w,This parameter is related to current iter.
        The parameter,type，decide which method to choose.In our progress, we test some method prposed by several articles.
        So the Chinese tips describe the name of the method in order to help us to recognize different method.
        :param iter_num:  the current iter
        :param type:  type of method
        :return: do not need return anything,just adjust w
        '''
        if type == 1:
            self.w = pso_inertia.inertia_01
        elif type == 2:
            self.w = pso_inertia.inertia_02
        elif type == 3:
            self.w = pso_inertia.inertia_03
        elif type == 4:
            self.w = pso_inertia.inertia_04
        elif type == 5:
            self.w = pso_inertia.inertia_05
        elif type ==  6:
            self.w = pso_inertia.inertia_06
        else :
            self.w = pso_inertia.inertia

    def cal_c(self,type):
        '''
        This function is used to adjust accelerate coefficients.
        There are several method which proposed by some articles to adjust accelerate coefficients.
        Our progress will test the performance of this method.
        We can distinct different method by parameter type of this function.
        iter parameter is necessary to calculate accelerate coefficientss.
        Before update vecolity ,we ought to use this method beacuse accelerate coefficients are the conponent of part2 and part3 of the v_update formula
        :param iter:
        :param type:
        :return:
        '''
        if type=="原始形态" or 0:
            self.cp = 2
            self.cg = 2

    def update_V(self):
        '''
        This function is used to update particles V for each particle.
        :return:
        '''
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim) # pop*dim个【0，1】的随机数

        self.V = self.w * self.V \
                 + self.cp * r1 * (self.pbest_x - self.X) \
                 + self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        '''
        This function is used to update particles X for each particle.
        :return:
        '''
        self.X = self.X + self.V
        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)


    def cal_y(self):
        '''
        calculate fitness Y for every x in X
        每一个pop都会对应一个适应值存储到Y中
        :return:
        '''
        self.Y = self.func(self.X).reshape(-1, 1)
        #print(self.X)

        # 通过reshape(-1,1)将其转化维1列即pop*1规模
        return self.Y

    def update_pbest(self):
        '''
        更新个体最优：update personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        更新全局最优 update global best
        :return:
        '''
        if self.gbest_y > self.Y.min():
            #找到最小值所在的行标，切片拿出该粒子的位置，拷贝给新的全局最优位置
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()
        #print(self.gbest_y)

    def recorder(self):
        '''
        This function is used to racord some essential result data
        :return:
        '''
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self,w_type,c_type):
        '''
        main part
        :param w_type:
        :param ctype:
        :return:
        '''
        self.max_iter = self.max_iter
        for iter_num in range(self.max_iter):
            self.iter_num = iter_num+1
            self.cal_w(w_type)  #更新w
            self.cal_c(c_type)  # 更新加速因子c1,c2
            self.update_V()       # 更新速度
            self.recorder()     # 做记录
            self.update_X()     # 更新X即位置
            self.cal_y()        # 计算新一代的适应值
            self.update_pbest() # 更新粒子的历史最优
            self.update_gbest() # 更新全体的历史最优
            self.gbest_y_hist.append(self.gbest_y)

        print(self.gbest_y_hist)
        return self


