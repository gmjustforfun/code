import numpy as np
from sko.tools import func_transformer
from .base import SkoBase
from .operators import pso_inertia

class APSO(SkoBase):
    def __init__(self, func, dim=10, pop=40, max_iter=1000, lb=None, ub=None, w=0.8, c1=2, c2=2):
        self.func = func_transformer(func)  # 格式化后的测试函数即适应值函数
