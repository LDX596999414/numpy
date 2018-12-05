import numpy as np

#手动创建数组

#a = np.array([[0,1,2,3],[4,5,6,7]]) #4*2 array
#a = np.array([1,2,3,4,5,6])
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7]])
print(a)       #输出数组a
print(a.ndim)  #维度
print(a.shape) #数组形状       (4, 3)   4行 3列

#函数创建数组

b = np.arange(10) #包左不包右
print(b)         #[0 1 2 3 4 5 6 7 8 9]

#范围内+3

b2 = np.arange(1,9,3)       #(start,end,step)
print(b2)                   #[1 4 7]

#范围等分

b3 = np.linspace(0,10,5)   #（start,end, num-point）  等距分割
print(b3)                  #[ 0.   2.5  5.   7.5 10. ]

#初始化为1的矩阵

b4 = np.ones((3,3)) # 1 3*3
print(b4)
'''
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
 '''
#初始化为0的矩阵
b5 = np.zeros((2,2)) # 0 2*2
print(b5)
'''
[[0. 0.]
 [0. 0.]]
'''

#单位矩阵

b6 = np.eye(3)
print(b6)
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
'''

#对角线数组

b7 = np.diag([1,2,3,4])
print(b7)
'''
[[1 0 0 0]
 [0 2 0 0]
 [0 0 3 0]
 [0 0 0 4]]
'''

#为空的数组

b8 = np.empty(0)
print(b8)  #[]



#切片和索引

d = np.diag(np.arange(3))
print(d)
'''
[[0 0 0]
 [0 1 0]
 [0 0 2]]
'''

print(d[1],d[1,1],d[2,1])    # third line, second column
#[0 1 0] 1 0
# In 2D, the first dimension corresponds to rows, the second to columns.

d2 = np.arange(10)
print(d2)  #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(d2[2:9:3]) #[start:end:step]
#array([2, 5, 8])
print(d2[::-1]) # [9 8 7 6 5 4 3 2 1 0]  倒序
#d[strat:stop:steps] => [strat stop) ++steps
'''
d2[:4]    #[0,1,2,3]
d2[::3]   #[0,3,6,9]
d2[3:]    #[3,4,5,6,7,8,9]
'''

#练习 对角线上递增矩阵实现左下平移一个单位

d3 = np.diag(np.array([1,2,3,4,5,6]))
'''
 [[1 0 0 0 0 0]
 [0 2 0 0 0 0]
 [0 0 3 0 0 0]
 [0 0 0 4 0 0]
 [0 0 0 0 5 0]
 [0 0 0 0 0 6]]
'''
d3[:,:5]=d3[:,1:]  #发现“=”赋值是直接进行交换，并不是copy赋值
'''
 [[0 0 0 0 0 0]
  [2 0 0 0 0 0]
  [0 3 0 0 0 0]
  [0 0 4 0 0 0]
  [0 0 0 5 0 0]
  [0 0 0 0 6 6]]
'''
d3[:,::-1]
d3[d3>3] = 100
d3[5,5]=0
print(d3)
'''
[[0 0 0 0 0 0]
 [2 0 0 0 0 0]
 [0 3 0 0 0 0]
 [0 0 4 0 0 0]
 [0 0 0 5 0 0]
 [0 0 0 0 6 0]]
'''
np.stack()
np.hstack()
np.dstack()
np.concatenate()
np.tile()
np.newaxis()
np.repeat()
np.where()
np.argmax()
np.dot()
np.transpose()











