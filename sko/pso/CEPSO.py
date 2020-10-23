import numpy as np
from sko.tools import func_transformer
from base import SkoBase
from operators.pso_inertia import inertia_01  # 引入线性递减的惯量权重调整策略
import math
import random

"""
CEPSO 协同进化粒子群优化算法
CEPSO  is proposed in "Distributed Co-evolutionary Particle Swarm Optimization Using Adaptive Migration Strategy"
采用了多种群策略，自适应迁移策略
"""


class CEPSO(SkoBase):
    def __init__(self, func, dim, pop01=20, pop02=20, max_iter=150, lb=None, ub=None, w=0.9, c1=2, c2=2,
                 acceptance=0.01):
        """

        :param func:
        :param dim:
        :param pop01:
        :param pop02:
        :param max_iter:
        :param lb:
        :param ub:
        :param w:
        :param c1:
        :param c2:
        :param acceptance:
        """
        self.func = func_transformer(func)
        self.dim = dim
        self.pop01 = pop01
        self.pop02 = pop02
        self.max_iter = max_iter
        self.current_iter = 0
        self.w = w
        self.cp = c1
        self.cg = c2
        self.acceptance  = acceptance

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        # 分别为GPSO和LPSO初始化两个种群
        # 初始化粒子的位置
        self.X1 = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop01, self.dim))
        self.X2 = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop02, self.dim))
        v_high = self.ub - self.lb
        # 初始化粒子的速度
        self.V1 = np.random.uniform(low=-v_high, high=v_high, size=(self.pop01, self.dim))  # speed of particles
        self.V2 = np.random.uniform(low=-v_high, high=v_high, size=(self.pop02, self.dim))  # speed of particles
        self.Y1 = self.cal_y(0)                                         # y = f(x) for all particles
        self.Y2 = self.cal_y(1)                                         # y = f(x) for all particles
        # 个最优为当前位置
        self.pbest_x1 = self.X1.copy()                                  # personal best location of every particle in history
        self.pbest_y1 = self.Y1.copy()                                  # best image of every particle in history
        self.pbest_x2 = self.X2.copy()                                  # personal best location of every particle in history
        self.pbest_y2 = self.Y2.copy()                                  # best image of every particle in history
        # 全局最优为初始的群体最优位置
        '----------------------局部版本的PSO的邻域怎么处理'
        self.local_size = 1                                             # 设局部的领域大小： 2 表示每个粒子之和他左右两个粒子相邻，
        # 把pop_size个粒子的索引看做他们在拓扑结构中的编号，如果一个粒子的编号为0，他的左边为pop_size-1,右半边为1
        # 每一个粒子都有一个局部最优的位置
        self.gbest_x1 = np.zeros((1, self.dim))                         # global best location for all particles
        self.gbest_y1 = np.inf                                          # global best y for all particles
        self.gbest_x2 = np.zeros((1, self.dim))                         # global best location for all particles
        self.gbest_y2 = np.inf                                          # global best y for all particles
        self.update_gbest()                                             # 评估当前全体最优粒子
        self.gbestY = np.inf
        self.gbestX = []
        self.lbest  = self.update_lbest()
        # 记录历史最优
        self.gbest_y1_hist = []                                         # gbest_y of every iteration

        # record verbose values
        self.record_mode = True
        self.record_value = {'X1': [], 'V1': [], 'Y1': [],'X2': [], 'V2': [], 'Y2': [],
                             'w': [], 'pop01': [], 'pop02': [],'gbest_GPSO':[],'gbest_LPSO':[],'gbest_All':[]}

    def cal_y(self, method=0):
        """
        calculate Y and update Y
        :param: method =1 局部PSO
        """

        if method == 0:
            self.Y1 = self.func(self.X1).reshape(-1, 1)  # calculate y for every x in X
            return self.Y1
        else:
            self.Y2 = self.func(self.X2).reshape(-1, 1)  # calculate y for every x in X
            return self.Y2

    def update_pbest(self):
        """
        update personal best，include pbest_x and pbest_y
        :return:
        """
        # 更新每个粒子的个体最优值的位置和相应的适应值，越小越优

        self.pbest_x1 = np.where(self.pbest_y1 > self.Y1, self.X1, self.pbest_x1)
        self.pbest_y1 = np.where(self.pbest_y1 > self.Y1, self.Y1, self.pbest_y1)
        self.pbest_x2 = np.where(self.pbest_y2 > self.Y2, self.X2, self.pbest_x2)
        self.pbest_y2 = np.where(self.pbest_y2 > self.Y2, self.Y2, self.pbest_y2)

    def update_gbest(self):
        """
        update_gbest
        :return:
        """
        if self.gbest_y1 > self.Y1.min():  # 更新子群1全局最优的位置和适应值
            self.gbest_x1 = self.X1[self.Y1.argmin(), :].copy()
            self.gbest_y1 = self.Y1.min()

        if self.gbest_y2 > self.Y2.min():  # 更新子群2全局最优的位置和适应值
            self.gbest_x2 = self.X2[self.Y2.argmin(), :].copy()
            self.gbest_y2 = self.Y2.min()
        # 更新全体最优的位置和速度
        if  self.gbest_y1< self.gbest_y2:
            self.gbestY = self.gbest_y1
            self.gbestX = self.gbest_x1
        else:
            self.gbestY = self.gbest_y2
            self.gbestX = self.gbest_x2

    def update_lbest(self):
        """
        更新种群2的粒子的局部最优位置
        :return:
        """
        pop_size = self.pop02
        lbest = np.zeros((pop_size,self.dim))

        # lbest = [[]]
        for i in range(pop_size):  # 对每一个；粒子，存储他的领域
            neighbor_per = np.zeros(3)
            neighbor_fit = np.zeros(3)
            if i==0:

                neighbor_per[0] = pop_size - 1
                neighbor_per[1] = i
                neighbor_per[2] = 1
                neighbor_fit[0] = self.Y2[(pop_size - 1)]
                neighbor_fit[1] = self.Y2[i]
                neighbor_fit[2] = self.Y2[1]
                index = int(neighbor_per[np.argmin(neighbor_fit)])

                lbest[i] = self.X2[index]
            elif i==(pop_size-1):
                neighbor_per[0] = pop_size - 2
                neighbor_per[1] = i
                neighbor_per[2] = 0
                neighbor_fit[0] = self.Y2[pop_size - 2]
                neighbor_fit[1] = self.Y2[i]
                neighbor_fit[2] = self.Y2[2]
                index = int(neighbor_per[np.argmin(neighbor_fit)])

                lbest[i] = self.X2[index]

            else:
                neighbor_per[0] = i - 1
                neighbor_per[1] = i
                neighbor_per[2] = i + 1
                neighbor_fit[0] = self.Y2[i - 1]
                neighbor_fit[1] = self.Y2[i]
                neighbor_fit[2] = self.Y2[i + 1]
                index = int(neighbor_per[np.argmin(neighbor_fit)])

                lbest[i] = self.X2[index]
        return lbest

    def update_v(self):
        """
        更新两个种群的速度
        :return:
        """

        r1 = np.random.rand(self.pop01, self.dim)
        r2 = np.random.rand(self.pop01, self.dim)
        r11 = np.random.rand(self.pop02, self.dim)
        r21 = np.random.rand(self.pop02, self.dim)

        self.V1 = self.w * self.V1 + \
                 self.cp * r1 * (self.pbest_x1 - self.X1) + \
                 self.cg * r2 * (self.gbest_x1 - self.X1)
        self.V2 = self.w * self.V2 + \
                  self.cp * r11 * (self.pbest_x2 - self.X2) + \
                  self.cg * r21 * (self.update_lbest() - self.X2)


    def update_x(self):
        """
        更新两个种群的位置
        :return:
        """

        self.X1 = self.X1 + self.V1
        self.X2 = self.X2 + self.V2
        if self.has_constraints:
            self.X1 = np.clip(self.X1, self.lb, self.ub)
            self.X2 = np.clip(self.X2, self.lb, self.ub)

    def recorder(self):
        """
        record data of every generation
        :return:
        """
        if not self.record_mode:
            return
        self.record_value['X1'].append(self.X1)  # 每一代记录一个X1，shape(particles_num,dim)
        self.record_value['V1'].append(self.V1)  # 每一代记录一个V1，shape(particles_num,dim)
        self.record_value['Y1'].append(self.Y1)  # 每一代记录一个Y1，shape(particles_num,dim)
        self.record_value['X2'].append(self.X2)  # 每一代记录一个X2，shape(particles_num,dim)
        self.record_value['V2'].append(self.V2)  # 每一代记录一个V2，shape(particles_num,dim)
        self.record_value['Y2'].append(self.Y2)  # 每一代记录一个Y2，shape(particles_num,dim)
        self.record_value['w'].append(self.w)  # 每一代记录一个w，shape(particles_num,dim)
        self.record_value['pop01'].append(self.pop01)  # 每一代记录一个种群规模，shape(max_iter,1)
        self.record_value['pop02'].append(self.pop02)  # 每一代记录一个种群规模，shape(max_iter,1)
        self.record_value['gbest_GPSO'].append(self.gbest_y1)  # 每一代记录一个子种群最优，shape(max_iter,1)
        self.record_value['gbest_LPSO'].append(self.gbest_y2)  # 每一代记录一个种群最优，shape(max_iter,1)
        self.record_value['gbest_All'].append(self.gbestY)  # 每一代记录一个Y，shape(max_iter,1)

    def AMS(self):
        """
        ANS做迁移
        :return:
        """

        #  s1 计算两个种群的平均适应值
        # 比较大小，区分较优子群和较差子群
        pm = 0.01 + 0.99 * (math.exp(10*self.current_iter/self.max_iter)-1)/(math.exp(10)-1)
        if np.mean(self.Y1) < np.mean(self.Y2): # Y2是较差种群
            if np.random.rand()<pm and self.pop02>5:
                wait_particle_index = np.argmax(self.Y2)
                self.pop02 -= 1
                self.pop01 += 1
                wait_particle_position = self.X2[wait_particle_index]  # 找到待迁移的粒子的位置
                wait_particle_velocy = self.V2[wait_particle_index]  # 找到待迁移的粒子的位置
                "-------------------------------------------------"
                # 改变子种群的位置和速度
                X2 = self.X2.tolist()  # 转为list
                del X2[wait_particle_index]
                self.X2 =  np.array(X2)

                V2 = self.V2.tolist()  # 转为list
                del V2[wait_particle_index]
                self.V2 = np.array(V2)

                X1 = self.X1.tolist()
                X1.append(wait_particle_position)
                self.X1 = np.array(X1)

                V1 = self.V1.tolist()
                V1.append(wait_particle_velocy)
                self.V1 = np.array(V1)

                # 改变子种群的个体最优pbest_y 和个体适应值Y
                wait_particle_pbest_y = self.pbest_y2[wait_particle_index]
                pbest_y2 = self.pbest_y2.tolist()
                del pbest_y2[wait_particle_index]
                self.pbest_y2 = np.array(pbest_y2)
                pbest_y1 = self.pbest_y1.tolist()
                pbest_y1.append(wait_particle_pbest_y)
                self.pbest_y1 = np.array(pbest_y1)

                wait_particle_Y = self.Y2[wait_particle_index]
                Y2 = self.Y2.tolist()
                del Y2[wait_particle_index]
                self.Y2 = np.array(Y2)
                Y1 = self.Y1.tolist()
                Y1.append(wait_particle_Y)
                self.Y1 = np.array(Y1)

                wait_particle_pbest_x = self.pbest_x2[wait_particle_index]
                pbest_x2 = self.pbest_x2.tolist()
                del pbest_x2[wait_particle_index]
                self.pbest_x2 = np.array(pbest_x2)
                pbest_x1 = self.pbest_x1.tolist()
                pbest_x1.append(wait_particle_pbest_x)
                self.pbest_x1 = np.array(pbest_x1)

                # print('经过迁移后，2 种群粒子数减少，他的位置shape: ', self.X2.shape)
                # print('经过迁移后，1 种群粒子数增加，他的位置shape: ', self.X1.shape)
                self.update_pbest()
                self.update_gbest()

                self.update_lbest()

        else: # Y1是较差种群
            if np.random.rand() < pm and self.pop01>5:
                wait_particle_index = np.argmax(self.Y1)
                self.pop01 -= 1
                self.pop02 += 1

                wait_particle_position = self.X1[wait_particle_index]  # 找到待迁移的粒子的位置
                wait_particle_velocy = self.V1[wait_particle_index]  # 找到待迁移的粒子的位置

                X1 = self.X1.tolist()  # 转为list
                del X1[wait_particle_index]
                self.X1 = np.array(X1)

                V1 = self.V1.tolist()  # 转为list
                del V1[wait_particle_index]
                self.V1 = np.array(V1)

                X2 = self.X2.tolist()
                X2.append(wait_particle_position)
                self.X2 = np.array(X2)

                V2 = self.V2.tolist()
                V2.append(wait_particle_velocy)
                self.V2 = np.array(V2)

                # 改变子种群的个体最优pbest_y 和个体适应值Y
                wait_particle_pbest_y = self.pbest_y1[wait_particle_index]
                pbest_y1 = self.pbest_y1.tolist()
                del pbest_y1[wait_particle_index]
                self.pbest_y1 = np.array(pbest_y1)
                pbest_y2 = self.pbest_y2.tolist()
                pbest_y2.append(wait_particle_pbest_y)
                self.pbest_y2 = np.array(pbest_y2)

                wait_particle_Y = self.Y1[wait_particle_index]
                Y1 = self.Y1.tolist()
                del Y1[wait_particle_index]
                self.Y1 = np.array(Y1)
                Y2 = self.Y2.tolist()
                Y2.append(wait_particle_Y)
                self.Y2 = np.array(Y2)

                wait_particle_pbest_x = self.pbest_x1[wait_particle_index]
                pbest_x1 = self.pbest_x1.tolist()
                del pbest_x1[wait_particle_index]
                self.pbest_x1 = np.array(pbest_x1)
                pbest_x2 = self.pbest_x2.tolist()
                pbest_x2.append(wait_particle_pbest_x)
                self.pbest_x2 = np.array(pbest_x2)
                # print('经过迁移后，1 种群粒子数减少，他的位置shape: ', self.X1.shape)
                # print('经过迁移后，2 种群粒子数增加，他的位置shape: ', self.X2.shape)
                self.update_pbest()
                self.update_gbest()
                self.update_lbest()


    def run(self):
         for epoch in range(self.max_iter):
             self.current_iter = epoch
             if epoch != 0:
                self.w = inertia_01(epoch,self.max_iter)        # LDW法更新w
             self.update_v()                        # 更新速度
             self.update_x()                        # 更新位置
             self.Y1 = self.cal_y(0)                # 计算适应值
             self.Y2 = self.cal_y(1)
             self.update_pbest()                    # 更新个体最优
             self.update_gbest()                    # 更新全体最优
             self.update_lbest()

             self.AMS()                             # 做种群迁移

             self.recorder()                        # 一代结束，做数据记录
             print(
                 'iter time : %d   w : %.8f%%  c1 : %.8f%%   c2 : %.8f%% ' % (self.current_iter, self.w, self.cp, self.cg))
             print(self.gbestY)