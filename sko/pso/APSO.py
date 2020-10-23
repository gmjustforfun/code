
import numpy as np
from sko.tools import func_transformer
from base import SkoBase
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
    x1, x2 = x
    return (x1-5)**2+(x2-5)**2




class APSO(SkoBase):
    def __init__(self, func, dim=30, pop=100, max_iter=100, lb=None, ub=None, w=0.9, c1=2, c2=2, acceptance=0.01):

        self.delta = np.random.uniform(high=0.1, low=0.05, size=1)
        self.func = func_transformer(func)
        self.w = w                      # inertia
        self.w_min = 0.4
        self.w_max = 0.9
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
        self.V = np.random.uniform(low=-v_high*0.2, high=v_high*0.2, size=(self.pop, self.dim))  # speed of particles
        self.Y = self.cal_y()           # y = f(x) for all particles
        self.pbest_x = self.X.copy()    # personal best location of every particle in history
        self.pbest_y = self.Y.copy()    # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf           # global best y for all particles
        self.gbest_y_hist = []          # gbest_y of every iteration
        # 评估当前全体最优粒子
        self.update_gbest()
        self.bestX_idx = self.pbest_y.argmin()
        self.worstX_idx = self.pbest_y.argmax()

        # 实验2所需，计算进化因子
        self.d = self.cal_d()  # the mean distance of each particle i to all the other particles.
        self.f = self.cal_f()  # 计算
        # record verbose values
        self.record_mode = True
        self.record_value = {'X': [], 'V': [], 'Y': [], 'gbest_each_generation': [], 'f': []}

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V
        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_d(self):
        """
        计算每个粒子到其他所有粒子的平均距离 the mean distance of each particle i to all the other particles.
        """
        d = np.zeros(self.pop)
        for i in range(self.pop):  # 对每一个粒子
            sum = 0
            for j in range(self.pop):
                if j != i:
                    sum += np.sqrt(np.sum(np.square(self.X[i] - self.X[j])))
            d[i] = sum / (self.pop - 1)
        self.d = d
        return self.d  #shape (pop)

    def cal_y(self):
        self.Y = self.func(self.X).reshape(-1, 1)

        return self.Y

    def cal_f(self):
        """
        f = (dg - dmin) / (dmax - dmin)
        """
        dg = self.d[self.Y.argmin()]
        if (self.d.max() - self.d.min()) == 0:
            self.f = 0
        elif self.d.max==self.d.min==dg==0:
            self.f = 0
        else:
            self.f = (dg - self.d.min()) / (self.d.max() - self.d.min())
        if not (0 <= self.f <= 1):
            print('f=' + str(np.round(self.f, 3)) + ' 超出範圍[0, 1]!!!')
            if self.f > 1:
                self.f = 1
            elif self.f < 0:
                self.f = 0
        return self.f

    def update_pbest(self):
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)
        # 每次更新了粒子位置后就可以计算欧式距离d和进化因子f
        self.cal_d()
        self.cal_f()

    def update_gbest(self):
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
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


    def ESE_adaptiveParam(self):
        # step1 根据f估计状态state,对每一个f（一代只有一个）
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
        elif 0.8 < self.f <=1:
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
        elif 0.6 < self.f <= 1:
            state[1] = 0
        # S3
        if 0 <= self.f <= 0.1:
            state[2] = 1
        elif 0.1 < self.f <= 0.3:
            state[2] = -5 * self.f + 1.5
        elif 0.3 < self.f <= 1:
            state[2] = 0
        # S4
        if 0 <= self.f <= 0.7:
            state[3] = 0
        elif 0.7 < self.f <= 0.9:
            state[3] = 5 * self.f - 3.5
        elif 0.9 < self.f <= 1:
            state[3] = 1

        if np.max(state) == np.min(state) == 0:
            print('因為f超出範圍[0, 1]，所以模糊推論異常!!!')

        # 统计了四种状态值
        state_target = state.argmax()

        self.w = 1 / (1 + 1.5 * np.exp(-2.6 * self.f))
        if not (self.w_min <= self.w <= self.w_max):
            print('w=' + str(np.round(self.w, 3)) + ' 超出範圍[0.4, 0.9]!!!')
            if self.w < self.w_min:
                self.w = self.w_min
            elif self.w > self.w_max:
                self.w = self.w_max

        if state_target == 0:  # S1 Exploration
            self.cp = self.cp + self.delta
            self.cg = self.cg - self.delta
        if state_target == 1:  # S2 Exploitation
            self.cp = self.cp + 0.5 * self.delta
            self.cg = self.cg - 0.5 * self.delta
        if state_target == 2:  # S3 Convergence
            self.cp = self.cp + 0.5 * self.delta
            self.cg = self.cg + 0.5 * self.delta

            self.ELS()

        if state_target == 3:
            self.cg = self.cp - self.delta
            self.cg = self.cg + self.delta
        if not (1.5 <= self.cp <= 2.5):
            # print('c1='+str(np.round(self.c1, 3))+' 超出範圍[1.5, 2.5]!!!')
            if self.cp < 1.5:
                self.cp = 1.5
            elif self.cp > 2.5:
                self.cp = 2.5
        if not (1.5 <= self.cg <= 2.5):
            # print('c2='+str(np.round(self.c2, 3))+' 超出範圍[1.5, 2.5]!!!')
            if self.cg < 1.5:
                self.cg = 1.5
            elif self.cg > 2.5:
                self.cg = 2.5

        # (12)
        if not (3 <= self.cp + self.cg <= 4):
            # print('c1='+str(np.round(self.c1, 3))+' + c2='+str(np.round(self.c2, 3))+' 超出範圍[3, 4]!!!')
            self.cp, self.cg = 4.0 * self.cp / (self.cp + self.cg), 4.0 * self.cg / (self.cp + self.cg)


    def ELS(self):
        rho = 1 - ((1 - 0.1) * self.iter_num / self.max_iter)  # 1. 生成随机数
        d = np.random.randint(low=0, high=self.dim, size=1)  # 生成随机位置

        P = self.gbest_x.copy()  # 全局最优粒子的位置向量
        X = self.X.copy()

        P[d] = P[d] + (self.ub[d] - self.lb[d]) * np.random.normal(loc=0.0, scale=rho ** 2, size=1)

        if P[d] > self.ub[d] : P[d] = self.ub[d]
        if P[d] < self.lb[d] : P[d] = self.lb[d]
        X[self.bestX_idx,:] = P.copy()
        score = self.func(X).reshape(-1,1)

        if min(score) < self.gbest_y:
            self.gbest_x = P.copy()
            self.gbest_y = min(score)
        else:
            # # case1. 原文版本
            self.X[self.worstX_idx, :] = P.copy()

            # case2. 我的版本
            # if min(score) < self.Y[self.worstX_idx]:
            #     self.X[self.worstX_idx, :] = P.copy()
            # if min(score) < np.max(self.pbest_y):
            #     idx = np.argmax(self.Y)
            #     self.Y[idx] = min(score)
            #     self.X[idx, :] = P.copy()

    def run(self, max_iter=None):
        flag = False
        self.bestX_idx = self.pbest_y.argmin()
        self.worstX_idx = self.pbest_y.argmax()
        self.max_iter = max_iter or self.max_iter
        arange = (i for i in range(self.max_iter))
        for iter_num in arange:
            self.iter_num = iter_num
            self.ESE_adaptiveParam()    # 为下一代的更新自适应调整参数w,c,c2
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
            # print('iter time : %d   w : %.8f%%  c1 : %.8f%%   c2 : %.8f%%  ' % (self.iter_num, self.w, self.cp, self.cg))
            # print(self.gbest_y)

        return self



