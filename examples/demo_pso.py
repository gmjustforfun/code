# %% Plot the result
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import openpyxl
from sko.PSO import PSO
"参数预设"
# [-100,100]
set_lb_01 =np.ones(10)*(-100)
set_ub_01 = np.ones(10)*100
# [-10,10]
set_lb_02 = np.ones(10)*(-10)
set_ub_02 = np.ones(10)*10
# [-30,30]
set_lb_03= np.ones(10)*(-30)
set_ub_03 = np.ones(10)*30
# [-1.28,1.28]
set_lb_04 = np.ones(10)*(-1.28)
set_ub_04 = np.ones(10)*1.28
# [-500,500]
set_lb_05 = np.ones(10)*(-500)
set_ub_05 = np.ones(10)*500
# [-5.12,5.12]
set_lb_06 = np.ones(10)*(-5.12)
set_ub_06 = np.ones(10)*5.12
# [-32,32]
set_lb_07 = np.ones(10)*(-32)
set_ub_07 = np.ones(10)*32

# [-100,100]
set_lb_011 = np.ones(30)*(-100)
set_ub_011 = np.ones(30)*100
# [-10,10]
set_lb_021 = np.ones(30)*(-10)
set_ub_021 = np.ones(30)*10
# [-30,30]
set_lb_031 = np.ones(30)*(-30)
set_ub_031 = np.ones(30)*30
# [-1.28,1.28]
set_lb_041 = np.ones(30)*(-1.28)
set_ub_041 = np.ones(30)*1.28
# [-500,500]
set_lb_051 = np.ones(30)*(-500)
set_ub_051 = np.ones(30)*500
# [-5.12,5.12]
set_lb_061 = np.ones(30)*(-5.12)
set_ub_061 = np.ones(30)*5.12
# [-32,32]
set_lb_071 = np.ones(30)*(-32)
set_ub_071 = np.ones(30)*32


"定义测试函数"
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
def demo_func_01_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    return x1 ** 2 + x2 ** 2 + x3 ** 2+ x4 ** 2+ x5 ** 2+ x6 ** 2+ x7 ** 2+ x8 ** 2+ x9 ** 2+ x10 ** 2
def demo_func_01_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    return x1 ** 2 + x2 ** 2 + x3 ** 2+ x4 ** 2+ x5 ** 2+ x6 ** 2+ x7 ** 2+ x8 ** 2+ x9 ** 2+ x10 ** 2+x11 ** 2+x12 ** 2+x13 ** 2+x14 ** 2+x15 ** 2+x16 ** 2+x17 ** 2+x18 ** 2+x19 ** 2+x20 ** 2+ x21 ** 2+x22 ** 2+x23 ** 2+x24 ** 2+x25 ** 2+x26 ** 2+x27 ** 2+x28 ** 2+x29 ** 2+x30 ** 2

def demo_func_02_10(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x
    return abs(x1)+ abs(x2)+ abs(x3)+ abs(x4)+ abs(x5)+ abs(x6)+ abs(x7)+ abs(x8)+ abs(x9)+ abs(x10)+abs(x1)*abs(x2)*abs(x3)*abs(x4)*abs(x5)*abs(x6)*abs(x7)*abs(x8)*abs(x9)*abs(x10)
def demo_func_02_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    return abs(x1)+ abs(x2)+ abs(x3)+ abs(x4)+ abs(x5)+ abs(x6)+ abs(x7)+ abs(x8)+ abs(x9)+ abs(x10)+abs(x1)*abs(x2)*abs(x3)*abs(x4)*abs(x5)*abs(x6)*abs(x7)*abs(x8)*abs(x9)*abs(x10)+abs(x11)+ abs(x12)+ abs(x13)+ abs(x14)+ abs(x15)+ abs(x16)+ abs(x17)+ abs(x18)+ abs(x19)+ abs(x20)+abs(x11)*abs(x12)*abs(x13)*abs(x14)*abs(x15)*abs(x16)*abs(x17)*abs(x18)*abs(x19)*abs(x20)+abs(x21)+ abs(x22)+ abs(x23)+ abs(x24)+ abs(x25)+ abs(x26)+ abs(x27)+ abs(x28)+ abs(x29)+ abs(x30)+abs(x21)*abs(x22)*abs(x23)*abs(x24)*abs(x25)*abs(x26)*abs(x27)*abs(x28)*abs(x29)*abs(x30)

def demo_func_03_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
    # print(x)
    # print(x1)
    # print("xx------------------------------------------")
    # print(xx)
    sum_total=0
    for i in range(10):
        sum = 0
        for j in range(i+1):
            sum += xx[j]
        sum = sum ** 2
        sum_total = sum_total+sum
    return sum_total
def demo_func_03_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30]
    sum_total = 0
    for i in range(30):
        sum = 0
        for j in range(i + 1):
            sum += xx[j]
        sum = sum ** 2
        sum_total = sum_total + sum
    return sum_total
def demo_func_04_10(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
    xx_1=xx.copy()
    for i in range(10):
        xx_1[i] = abs(xx[i])
    return max(xx_1)
def demo_func_04_30(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
          x25, x26, x27, x28, x29, x30]
    xx_1 = xx.copy()
    for i in range(30):
        xx_1[i] = abs(xx[i])
    return max(xx_1)
def demo_func_05_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    sum = 0
    for i in range(9):
        sum = sum + 100*((xx[i+1]-xx[i]**2)**2)+(xx[i]-1)**2
    return  sum
def demo_func_05_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
          x25, x26, x27, x28, x29, x30]
    sum = 0
    for i in range(29):
        sum = sum + 100*((xx[i + 1] - xx[i] ** 2) ** 2) + (xx[i] - 1) ** 2
    return sum
def demo_func_06_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    sum = 0
    for i in range(10):
        a = xx[i] + 0.5
        a1 = math.floor(a)
        a1 = a1**2
        sum = sum+a1
    return sum
def demo_func_06_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
          x25, x26, x27, x28, x29, x30]
    sum = 0
    for i in range(30):
        a = xx[i] + 0.5
        a1 = math.floor(a)
        a1 = a1 ** 2
        sum = sum + a1
    return sum
"注意：这里文章中的公式表述不明"
def demo_func_07_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    sum = 0
    for i in range(10):
        sum += (i+1)*(xx[i]**2)
    sum += np.random.rand(1)
    return sum
def demo_func_07_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
          x25, x26, x27, x28, x29, x30]
    sum = 0
    for i in range(30):
        sum += (i + 1) * (xx[i] ** 2)
    sum += np.random.rand(1)
    return sum
def demo_func_08_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    sum = 0
    for i in range(10):
        sum = sum+(-(xx[i]*math.sin(math.sqrt(abs(xx[i])))))
    return sum
def demo_func_08_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
          x25, x26, x27, x28, x29, x30]
    sum = 0
    for i in range(30):
        sum = sum + (-(xx[i] * math.sin(math.sqrt(abs(xx[i])))))
    return sum
def demo_func_09_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    sum = 0
    for i in range(10):
        sum += (xx[i]**2-10*math.cos(2*math.pi*xx[i])+10)
    return sum
def demo_func_09_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
          x25, x26, x27, x28, x29, x30]
    sum = 0
    for i in range(30):
        sum += (xx[i] ** 2 - 10 * math.cos(2 * math.pi * xx[i]) + 10)
    return sum
def demo_func_10_10(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    sum_01 = 0
    sum_02 = 0
    for i in range(10):
        sum_01 = sum_01+xx[i]**2
        sum_02 =sum_02+math.cos(2*math.pi*xx[i])
    sum_total = -20*math.exp(-0.2*math.sqrt((1/10)*sum_01))-math.exp((1/10)*sum_02)+20+math.e
    return sum_total
def demo_func_10_30(x):
    x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30 = x
    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
          x25, x26, x27, x28, x29, x30]
    sum_01 = 0
    sum_02 = 0
    for i in range(30):
        sum_01 = sum_01 + xx[i] ** 2
        sum_02 = sum_02 + math.cos(2 * math.pi * xx[i])
    sum_total = -20 * math.exp(-0.2 * math.sqrt((1 / 30) * sum_01)) - math.exp((1 / 30) * sum_02) + 20 + math.e
    return sum_total
'''
对10个函数10维和20维两种情况分别进行实验100次，存入100*20的矩阵中
'''

"10维，测试f1函数，"
pso_data = np.zeros((20,100)) # 存放两种维度下10个函数的100次实验的结果
mean_and_std = np.zeros((20,2)) #存放两种维度下10个函数的100次均值和标准差

for test_iter in range(100):
    pso = PSO(func=demo_func_01_10, dim=10, pop=20, max_iter=2000,
              lb=set_lb_01,ub=set_ub_01, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[0][test_iter]=pso.gbest_y


for test_iter in range(100):
    pso = PSO(func=demo_func_02_10, dim=10, pop=20, max_iter=2000,lb=set_lb_02,ub=set_ub_02, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[1][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_03_10, dim=10, pop=20, max_iter=2000,lb=set_lb_01,ub=set_ub_01, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[2][test_iter]=pso.gbest_y


for test_iter in range(100):
    pso = PSO(func=demo_func_04_10, dim=10, pop=20, max_iter=2000,lb=set_lb_01,ub=set_ub_01, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[3][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_05_10, dim=10, pop=20, max_iter=2000,lb=set_lb_03,ub=set_ub_03, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[4][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_06_10, dim=10, pop=20, max_iter=2000, lb=set_lb_01, ub=set_ub_01, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[5][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_07_10, dim=10, pop=20, max_iter=2000, lb=set_lb_04, ub=set_ub_04, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[6][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_08_10, dim=10, pop=20, max_iter=2000, lb=set_lb_05, ub=set_ub_05, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[7][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_09_10, dim=10, pop=20, max_iter=2000, lb=set_lb_06, ub=set_ub_06, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[8][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_10_10, dim=10, pop=20, max_iter=2000, lb=set_lb_07, ub=set_ub_07, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[9][test_iter]=pso.gbest_y

# "30维，测试f1函数"
for test_iter in range(100):
    pso = PSO(func=demo_func_01_30, dim=30, pop=20, max_iter=2000,
              lb=set_lb_011,ub=set_ub_011, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[10][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_02_30, dim=30, pop=20, max_iter=2000,lb=set_lb_021,ub=set_ub_021, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[11][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_03_30, dim=30, pop=20, max_iter=2000,lb=set_lb_011,ub=set_ub_011, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[12][test_iter]=pso.gbest_y


for test_iter in range(100):
    pso = PSO(func=demo_func_04_30, dim=30, pop=20, max_iter=2000,lb=set_lb_011,ub=set_ub_011, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[13][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_05_30, dim=30, pop=20, max_iter=2000,lb=set_lb_031,ub=set_ub_031, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[14][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_06_30, dim=30, pop=20, max_iter=2000, lb=set_lb_011, ub=set_ub_011, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[15][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_07_30, dim=30, pop=20, max_iter=2000, lb=set_lb_041, ub=set_ub_041, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[16][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_08_30, dim=30, pop=20, max_iter=2000, lb=set_lb_051, ub=set_ub_051, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[17][test_iter]=pso.gbest_y

for test_iter in range(100):
    pso = PSO(func=demo_func_09_30, dim=30, pop=20, max_iter=2000, lb=set_lb_061, ub=set_ub_061, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[18][test_iter]=pso.gbest_y


for test_iter in range(100):
    pso = PSO(func=demo_func_10_30, dim=30, pop=20, max_iter=2000, lb=set_lb_071, ub=set_ub_071, w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    pso_data[19][test_iter]=pso.gbest_y
# 按行求均值和标准差
# mean_and_std[:][0] = np.mean(pso_data, axis=1)
# mean_and_std[:][1] = np.std(pso_data, axis=1)
# print(pso_data[0])
# print(np.mean(pso_data[0]))
# plt.plot(pso.gbest_y_hist)
# plt.show()




# Do PSO
# pso = PSO(func=demo_func, dim=3, pop=40, max_iter=150, lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
# pso.run()
# print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
# plt.plot(pso.gbest_y_hist)
# plt.show()

# %% PSO without constraint:
# pso = PSO(func=demo_func, dim=3)
# fitness = pso.run()
# print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

wb = openpyxl.load_workbook('test.xlsx')
# create a Pandas Excel writer using xlswriter,这是写入一个已经存在的文件中
writer = pd.ExcelWriter('test.xlsx',engine='openpyxl')
writer.book = wb
df01 = pd.DataFrame(data=pso_data,index=['f1_10','f2_10','f3_10','f4_10','f5_10','f6_10','f7_10','f8_10','f9_10','f10_10','f1_30','f2_30','f3_30','f4_30','f5_30','f6_30','f7_30','f8_30','f9_30','f10_30'])
df01.to_excel(writer, sheet_name='方法四', startcol=0, index=False)
# df02 = pd.DataFrame(data=mean_and_std)
# df02.to_excel(writer, sheet_name='惯量权重线性递减法统计(LWD)', startcol=0, index=False)
writer.save() # 一定要保存

