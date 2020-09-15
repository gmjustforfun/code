import numpy as np
# %% Plot the result
import pandas as pd
import matplotlib.pyplot as plt
import math
# %%
from sko.GA import GA

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

def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)

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



# ga = GA(func=demo_func_01_30, n_dim=30, size_pop=100, max_iter=60000, lb=set_lb_011, ub=set_ub_011, precision=0.01)
# best_x, best_y = ga.run()
# print('best_x:', best_x, '\n', 'best_y:', best_y)

demo_func = lambda x: (x[0] - 1) ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=[-1, -1, -1], ub=[5, 1, 1], precision=[2, 1, 1e-7])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
