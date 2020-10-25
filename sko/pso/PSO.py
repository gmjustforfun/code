import numpy as np
from sko.tools import func_transformer
from sko.base import SkoBase

class PSO(SkoBase):
    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.9, c1=2, c2=2,acceptance=0.01):
        self.acceptance = acceptance    # 可接受精度
        self.func = func_transformer(func)
        self.w = w                      # inertia
        self.cp, self.cg = c1, c2       # parameters to control personal best, global best respectively
        self.pop = pop                  # number of particles
        self.dim = dim                  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter        # max iter
        self.iter_num = 0               # current iter no
        self.acceptance_iter = 0        # 达到精度的代数

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
        self.update_gbest()
        # record verbose values
        self.record_mode = True
        self.record_value = {'X': [], 'V': [], 'Y': [], 'gbest_each_generation': []}

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_w(self):
        return 0.9-0.5*self.iter_num/self.max_iter

    def update_X(self):
        self.X = self.X + self.V
        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        self.Y = self.func(self.X).reshape(-1, 1)  # calculate y for every x in X
        return self.Y

    def update_pbest(self):
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        if self.gbest_y > self.Y.min():  # 更新全局最优的位置和适应值
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)  # 每一代记录一个X，shape(particles_num,dim)
        self.record_value['V'].append(self.V)  # 每一代记录一个V，shape(particles_num,dim)
        self.record_value['Y'].append(self.Y)  # 每一代记录一个Y，shape(particles_num,dim)
        self.record_value['gbest_each_generation'].append(self.X[self.Y.argmin(), :])

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        arange = (i for i in range(self.max_iter))
        for iter_num in arange:
            self.iter_num = iter_num    # 记录档当前迭代次数
            self.w = self.update_w()    # 线性更新惯量权重
            self.update_V()             # 更新速度
            self.update_X()             # 更新位置
            self.cal_y()                # 计算适应值
            self.update_pbest()         # 更新粒子的历史最优
            self.update_gbest()         # 更新全体历史最优
            self.recorder()             # 每一代记录一次X,Y,V
            self.gbest_y_hist.append(self.gbest_y)
            # print('iter time : %d   w : %.8f%%  c1 : %.8f%%   c2 : %.8f%% ' % (self.iter_num, self.w, self.cp, self.cg))
            # print(self.gbest_y)
        return self