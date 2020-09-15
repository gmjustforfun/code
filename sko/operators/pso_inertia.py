import numpy as np

def inertia(self):
    '''
    不做修改的惯量权重法
    :param self:
    :return:
    '''
    self.w=self.w
    return self.w

def inertia_01(self):
    '''
    the first method of inertia
    the formula is come from the article "粒子群算法的惯量权重"研究LWD方法
    formula :
    :param self:
    :return: new w
    '''
    self.w = self.w_max - (self.w_max - self.w_min) / self.iter_num * self.max_iter
    return self.w

def inertia_02(self):
    '''
    the second method of inertia
    the formula is come from the article "粒子群算法的惯量权重"研究随机惯量权重方法
    formula :
    :param self:
    :return:new w
    '''
    self.w = np.random.uniform(low=0.4, high=0.6)
    return self.w

def inertia_03(self):
    '''
    the third method of inertia
    the formula is come from the article "粒子群算法的惯量权重"研究凹函数递减法方法
    formula :
    :param self:
    :return:new w
    '''
    self.w = -(self.w_max - self.w_min)(self.iter_num / self.max_iter) ** 2 + self.w_max
    return self.w

def inertia_04(self):
    '''
    the forth method of inertia
    the formula is come from the article "粒子群算法的惯量权重"研究凸函数递减方法
    fumula :
    :param self:
    :return:new w
    '''
    self.w = (self.w_max - self.w_min)(self.iter_num / self.max_iter - 1) ** 2 + self.w_min
    return self.w

def inertia_05(self):
    '''
    the fifth method of inertia
    the formula is come from the article "Self-organizing hierarchical particle swarm optimizer with time-varying acceleration coefficients"
    提出的线性递减法LWD，这个公式和詹老师在文献中给出的有细微差别
    formula : w = (w_max-w_min)(ITERMAX-iter)/ITERMAX+w_min
    :param self:
    :return:new w
    '''
    self.w = (self.w_max - self.w_min)(self.max_iter - self.iter_num) / self.max_iter + self.w_min
    return self.w

def inertia_06(self):
    '''
    the sixth method of inertia
    the formula is come from the article "Self-organizing hierarchical particle swarm optimizer with time-varying acceleration coefficients"
    /提出的随机惯量递减法RANDIW，这个公式和詹老师在文献中给出的有细微差别
    w = 0.5+rand(0,1)/2
    :param self:
    :return:new w
    '''
    self.w = 0.5 + np.random.uniform(low=0, high=1)/2
    return self.w

