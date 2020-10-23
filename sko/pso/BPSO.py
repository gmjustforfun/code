
import numpy as np
from sko.tools import func_transformer
from sko.base import SkoBase


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

    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.9, c1=2, c2=2,acceptance=0.01):
        '''
        contruct function
        :param func:        目标函数
        :param dim:         维度
        :param pop:         粒子数目
        :param max_iter:    最大迭代次数
        :param lb:          上界
        :param ub:          下界
        :param w:           惯量权重
        :param c1:          加速系数1
        :param c2:          加速系数2
        :param acceptance   可接受的精度
        '''
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
        # 评估当前全体最优粒子
        self.update_gbest()
        # record verbose values
        self.record_mode = True
        self.record_value = {'X': [], 'V': [], 'Y': [], 'gbest_each_generation': []}

    def update_V(self):
        """
        update V
        :return:
        """
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
        # 更新每个粒子的个体最优值的位置和相应的适应值，越小越优
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        """
        update_gbest;更新全局最优的粒子的位置和对应的适应值
        :return:
        """
        if self.gbest_y > self.Y.min():  # 更新全局最优的位置和适应值
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        """
        record data of every generation
        :return:
        """
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)  # 每一代记录一个X，shape(particles_num,dim)
        self.record_value['V'].append(self.V)  # 每一代记录一个V，shape(particles_num,dim)
        self.record_value['Y'].append(self.Y)  # 每一代记录一个Y，shape(particles_num,dim)
        # 记录每一代的全局最优的位置 gbest_X,(particles_num,dim)
        self.record_value['gbest_each_generation'].append(self.X[self.Y.argmin(), :])

    def run(self, max_iter=None):

        self.max_iter = max_iter or self.max_iter
        flag = False
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

            if flag == False and self.gbest_y <= self.acceptance:
                flag = True
                self.acceptance_iter = iter_num
            self.gbest_y_hist.append(self.gbest_y)
            # print('iter time : %d   w : %.8f%%  c1 : %.8f%%   c2 : %.8f%% ' % (self.iter_num, self.w, self.cp, self.cg))
            # print(self.gbest_y)
        return self

