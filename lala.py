import numpy as np
# a='aa'
# b=[]
# print(a) if b is None else print(b)
# print(np.ones(3))# [1. 1. 1.]
# print(-np.ones(3)) # [-1. -1. -1.]
# print(np.array([0, -1, 0.5]))
# a=1
# b=1
# c=1
# d=2
# assert a==1==b==c,'a==1==b==c'
# print(np.random.uniform(low=[0, -1, 0.5], high=[1,1,1], size=(10, 3)))
# 生成10*3，10为粒子个数，3为维度，且每一列都对应者变量的上下界
# [[ 0.89903499  0.89598783  0.73369037]
#  [ 0.42294917  0.50246091  0.56962041]
#  [ 0.66026545 -0.52193213  0.69854674]
#  [ 0.86574994 -0.84523039  0.66202131]
#  [ 0.52203548  0.79777535  0.91576257]
#  [ 0.33843875  0.46122255  0.98217009]
#  [ 0.14776705 -0.38980091  0.61932422]
#  [ 0.41854278  0.08875426  0.90521417]
#  [ 0.90437233 -0.23452748  0.87213468]
#  [ 0.698656   -0.37286464  0.82091155]]
# print(np.ones(3)*10)
#
# print(np.random.rand(10, 3))
# print( np.random.uniform(low=0.4,high=0.6))
# a=1
# if a=='nihao' or 1:
#     print("niahisadha")
# for i in range(1):
#     print(i)
#
# data = [ 95.26400481,-56.67224248,41.12458172,-90.7455364,24.03254827,35.15270597,42.31895008,7.65509617,-96.07924084,21.26398857]
# sum = 0
# for i in range(10):
#     sum+=data[i]**2
# print(sum)
# #35559.10515511
# a=[1,2,3,3,4]
# print(max(a))
#
#
# print(2**2)
# print(np.random.rand(1))
# a = np.zeros(10)
# print(np.zeros(10))
# print(np.mean(a))
p = np.random.randint(low=0, high=2, size=(3, 10))
print(p)
print(p[:, :10])

numerical = int("111111110011001",2)  # 转化为十进制数值
print(numerical)
print(-100 +numerical*(100-(-100))/(2**15-1))
numerical = int("111111110011001",2)  # 转化为十进制数值

# 1 1 1 1 1 1 1 1 0 0 1 1 0 0 1
# 0.66460158

a = [500, 1000, 1500, 2000]

b = [str(i) for i in a]

print(a)
print(b)