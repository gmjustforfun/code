import numpy as np

# V1 = np.random.uniform(low=-10, high=10, size=(5, 3))
# print(type(V1))
# print(V1)
#
# print("------------------------------")
# V2 = V1.tolist()
# print(type(V2))
# del V2[0]
# print(V2)
# V1 = np.array(V2)
# print(type(V1))
# print(V1)

v1 = np.random.uniform(low=-10, high=10, size=(5, 3))
print(v1)
wait = [1,2,3]
v2 = v1.tolist()
v2.append(wait)
print(v2)
v1 = np.array(v2)
print(v1)
