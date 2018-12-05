print("计算1到100累加：")
result = 0
for i in range(101):
    result+=i
print(result)
print(i)
#################################################3
import numpy as np
a = np.random.randint(1,8,size=(3,4,2,1))
x = np.random.randint(1,8,size=(2,3,4))                  #randint(low, high=None, size=None, dtype='l')
print(x.shape)
y = x[:,np.newaxis,:,:]
print(y)        #   (2, 1, 3, 4)










