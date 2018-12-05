import numpy as np

a1 = np.array([[1,2],[3,4]])
a2 = np.array([[5,6],[7,8]])
print(a1)
print(np.hstack([a1,a2]))   #横向合并
'''
[[1 2 5 6]
 [3 4 7 8]]
'''
print(np.vstack([a1,a2]))  #纵向合并
'''
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
'''
# 矩阵的合并也可以通过concatenate方法。
'''
np.concatenate( (a1,a2), axis=0 )  等价于  np.vstack( (a1,a2) )       #0行，1列
np.concatenate( (a1,a2), axis=1 )  等价于  np.hstack( (a1,a2) )
'''
print(np.concatenate((a1,a2),axis=0))
print(np.concatenate((a1,a2),axis=1))

print(a1.T)  #矩阵的转置
'''
[[1 3]
 [2 4]]
'''
#矩阵的最值
a3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
"""
'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

print(a3.max())
print(a3.min())
print(a3.max(axis=0))  # axis=0 行方向最大（小）值，即获得每列的最大（小）值
print(a3.min(axis=1))  # axis=1 列方向最大（小）值，即获得每行的最大（小）值
print(a3.argmax())     # 要想获得最大最小值元素所在的位置，可以通过argmax函数来获得  8
print(a3.argmin())     #0
print(a3.mean())       #平均值 ，同样地，可以通过关键字axis参数指定沿哪个方向获取平均值
print(a3.sum())        #求和 45
"""

#建立矩阵
b1 = np.arange(15).reshape(3,5)
'''
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
'''
#矩阵的乘法
b2 = np.arange(9).reshape(3,3)
'''
[[0 1 2]
 [3 4 5]
 [6 7 8]]
'''
b3 = np.array([[1,4,7],[2,5,8],[3,6,9]])
'''
[[1 4 7]
 [2 5 8]
 [3 6 9]]
'''
print(np.dot(b2,b3)) #对于二维矩阵，计算真正意义上的矩阵乘积，同线性代数中矩阵乘法的定义。
                     # 对于一维矩阵，计算两者的内积
                     # print(b2@b3)  等同于print(np.dot(b2,b3))
'''
[[  8  17  26]
 [ 26  62  98]
 [ 44 107 170]]
'''
b4 = np.arange(4)
b5 = b4[::-1]     #切片 Sliceing
print(b5)
print(np.dot(b4,b5)) #4
       #print(b4@b5)等同于print(np.dot(b4,b5))

######################根据条件生成新的数组######################
arr = np.random.randn(4,4)
print(arr)
print(np.where(arr>0,1,-1))   #     np.where(cond,xarr,yarr)

'''
p.where 函数是三元表达式 x if condition else y的矢量化版本
result = np.where(cond,xarr,yarr)
当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
栗子：假设有一个随机数生成的矩阵，希望将所有正值替换为2，负值替换为-2
'''
#########################重复#############################
x = np.arange(1, 5).reshape(2, 2)
'''
[[1 2]
 [3 4]]
'''
print(np.repeat(x, 2))
# 对数组中的每一个元素进行复制
# 除了待重复的数组之外，只有一个额外的参数时，高维数组也会 flatten 至一维
print(np.repeat(x,3,axis=1))   #我们在某一轴上进行复制，比如在行的方向上（axis=1），在列的方向上（axis=0）：
'''
[[1 1 1 2 2 2]
 [3 3 3 4 4 4]]
 '''
print(np.repeat(x,[2,1],axis=0))    #某一轴的方向上（axis=0/1），对不同的行/列复制不同的次数：
'''
[[1 2]
 [1 2]
 [3 4]]
'''
'''
help(np.repeat)
np.repeat(3, 4)  把三重复四遍
array([3, 3, 3, 3])
'''

print(np.tile(x,2))    #整体复制
'''
[[1 2 1 2]
 [3 4 3 4]]
'''
############对二维数组的transpose操作就是对原数组的转置操作
two=np.arange(16).reshape(4,4)
'''
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
'''
two.transpose()
'''
array([[ 0,  4,  8, 12],
       [ 1,  5,  9, 13],
       [ 2,  6, 10, 14],
       [ 3,  7, 11, 15]])

two.transpose(1,0)
array([[ 0,  4,  8, 12],
       [ 1,  5,  9, 13],
       [ 2,  6, 10, 14],
       [ 3,  7, 11, 15]])
'''
#原数组two的数组两个轴为（x，y），对应的下标为（0,1），np.transpose()传入的参数为（1,0），即将原数组的x,y轴互换。