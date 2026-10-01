import numpy as np

data = np.array([2,4,10,12,3,20,30,11,25])
M1 = [4]
M2 = [11]

x1 = abs(M1-data)
print(x1)

x2 = abs(M2-data)
print(x2)

cluster = [C1 if x1>x2 else C2]
print(cluster)