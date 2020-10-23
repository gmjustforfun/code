#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987


import numpy as np
from base import SkoBase
from abc import ABCMeta, abstractmethod
from operators import crossover, mutation, ranking, selection
from ga.GA import GeneticAlgorithmBase, GA

class DifferentialEvolutionBase(SkoBase, metaclass=ABCMeta):
    pass


class DE(GeneticAlgorithmBase):
    def __init__(self, func, n_dim, F=0.5,
                 size_pop=50, max_iter=200, prob_mut=0.3,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut,
                         constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
        self.prob_mut = prob_mut # 交叉概率
        self.F = F # 缩放因子，用于变异
        self.V, self.U = None, None
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.crtbp() # 生成初始种群

    def crtbp(self):
        '''
        create the population创建初始种群
        :return:
        '''
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        return self.X

    def chrom2x(self, Chrom):
        pass

    def ranking(self):
        pass

    def mutation(self):
        '''
        变异操作，即差分策略
        是随机选取种群中的两个不同的个体，将其向量差缩放后于待变异个体进行向量和乘
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        '''
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

        # 这里F用固定值，为了防止早熟，可以换成自适应值
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)
        return self.V

    def crossover(self):
        '''
        交叉操作，
        if rand < prob_crossover, use V, else use X
        '''
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        '''
        贪婪选择
        greedy selection
        '''
        X = self.X.copy()  # 拷贝当前代的X
        f_X = self.x2y().copy()  # 拷贝当前代的函数值
        self.X = U = self.U  # 把交叉后得到的种群赋给X
        f_U = self.x2y()   # 在计算当前的函数值

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)  # 比较，如果满足条件，则为X否则为交叉后的U
        return self.X

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        # 一次运行迭代max_iter代，每一代的适应值Y存放在all_history_Y中
        for i in range(self.max_iter):
            self.mutation() # 变异即差分
            self.crossover() # 交叉
            self.selection() #选择

            # record the best ones
            generation_best_index = self.Y.argmin() # 记录最优代
            self.generation_best_X.append(self.X[generation_best_index, :].copy()) # 记录最优代的X
            self.generation_best_Y.append(self.Y[generation_best_index])# 记录最优代的Y
            self.all_history_Y.append(self.Y)

        global_best_index = np.array(self.generation_best_Y).argmin()
        global_best_X = self.generation_best_X[global_best_index]
        global_best_Y = self.func(np.array([global_best_X]))
        return global_best_X, global_best_Y
