__version__ = '0.5.8'

from sko.pso import PSO
from . import DE, GA, SA, ACA, AFSA, IA


def start():
    print('''
    scikit-opt import successfully,
    version: {version}
    Author: Guo Fei,
    Email: guofei9987@foxmail.com
    repo: https://github.com/guofei9987/scikit-opt,
    documents: https://scikit-opt.github.io/
    '''.format(version=__version__))
