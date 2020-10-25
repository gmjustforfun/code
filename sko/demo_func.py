import numpy as np
from scipy import spatial
import math


def function_for_TSP(num_points, seed=None):
    if seed:
        np.random.seed(seed=seed)

    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points randomly
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    # print('distance_matrix is: \n', distance_matrix)

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    return num_points, points_coordinate, distance_matrix, cal_total_distance

# ok
def sphere(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put

def schs222(x):
    out_put = 0
    out_put01 = 1
    for i in x:
        out_put += abs(i)
        out_put01 = abs(i)*out_put01
    out_put = out_put01+out_put
    return out_put

def Schwefel_P222(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

def schls221(x):
    max = -np.inf
    for xi in x:
        if abs(xi) > max:
            max = abs(xi)
    return max

def schwefel(x):
    sum = 0
    for xi in x:
        sum += (-xi*np.sin(xi))
    return sum

def Step(x):
    sum = 0
    for xi in x:
        sum = sum + np.floor(xi+0.5) ** 2
    return sum

def schaffer(p):
    '''
    二维函数，具有无数个极小值点、强烈的震荡形态。很难找到全局最优值
    在(0,0)处取的最值0
    -10<=x1,x2<=10
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(np.sqrt(x))) - 0.5) / np.square(1 + 0.001 * x)

def shubert(p):
    '''
    2-dimension
    -10<=x1,x2<=10
    has 760 local minimas, 18 of which are global minimas with -186.7309
    '''
    x, y = p
    part1 = [i * np.cos((i + 1) * x + i) for i in range(1, 6)]
    part2 = [i * np.cos((i + 1) * y + i) for i in range(1, 6)]
    return np.sum(part1) * np.sum(part2)

def quadric(x):

    result = 0
    for i in range(len(x)):
        sum = 0
        for j in range(i+1):
            sum += x[j]
        sum = sum**2
        result += sum
    return result

def quadricNoise(x):
    sum = 0
    i = 0
    for xi in x:
        sum += i*xi**4
        i += 1
    result = sum + np.random.rand()
    return result

def ackley(x):
    part1 = 0
    part2 = 0
    for i in x:
        part1 += (i ** 2)
        part2 += np.cos(2 * np.pi * i)
    left = 20 * np.exp(-0.2 * ((part1 / x.shape[0]) ** .5))
    right = np.exp(part2 / x.shape[0])
    return -left - right + 20 + np.e

def griewank(p):
    '''
    存在多个局部最小值点，数目与问题的维度有关。
    此函数是典型的非线性多模态函数，具有广泛的搜索空间，是优化算法很难处理的复杂多模态问题。
    在(0,...,0)处取的全局最小值0
    -600<=xi<=600
    '''
    part1 = [np.square(x) / 4000 for x in p]
    part2 = [np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(p)]
    return np.sum(part1) - np.prod(part2) + 1

def rastrigrin(p):
    '''
    多峰值函数，也是典型的非线性多模态函数
    -5.12<=xi<=5.12
    在范围内有10n个局部最小值，峰形高低起伏不定跳跃。很难找到全局最优
    has a global minimum at x = 0  where f(x) = 0
    '''
    return np.sum([np.square(x) - 10 * np.cos(2 * np.pi * x) + 10 for x in p])

def rosenbrock(p):
    '''
    -2.048<=xi<=2.048
    函数全局最优点在一个平滑、狭长的抛物线山谷内，使算法很难辨别搜索方向，查找最优也变得十分困难
    在(1,...,1)处可以找到极小值0
    :param p:
    :return:
    '''
    n_dim = len(p)
    res = 0
    for i in range(n_dim - 1):
        res += 100 * np.square(np.square(p[i]) - p[i + 1]) + np.square(p[i] - 1)
    return res

def Schaffers_f6_function(p):
    x,y = p
    res = 0.5+((math.sin(x**2+y**2))**2-0.5)/((1+0.001*(x**2+y**2))**2)

if __name__ == '__main__':
    print(sphere((0, 0)))
    print(schaffer((0, 0)))
    print(shubert((-7.08350643, -7.70831395)))
    print(griewank((0, 0, 0)))
    print(rastrigrin((0, 0, 0)))
    print(rosenbrock((1, 1, 1)))
