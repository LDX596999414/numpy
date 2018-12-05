import numpy as np
import pandas as pd

a1 = np.array([[4,3],[2,1]])
a2 = np.arange(4).reshape(2,2)  #左行右列
print(a1)
print(a2)
print(a1*a2)   #对应位子元素相乘
print(a1@a2)
print(np.dot(a1,a2))
a3 = np.random.random((2,4))  # 2行4列 从0-1的随机数字  230881199501021738
print(a3)

#### np .split() 分割矩阵
A = np.arange(9).reshape(3,3)
print(np.array_split(A,4,axis=0)) #不能项分割
#[array([[0, 1, 2]]), array([[3, 4, 5]]), array([[6, 7, 8]]), array([], shape=(0, 3), dtype=int32)]

###############################################
B = A.copy()    # deep copy  只把A的值给B 两者不关联！
A[1, 1] = 0
print(A)
print(B)
'''
[[0 1 2]
 [3 0 5]
 [6 7 8]]
 
[[0 1 2]
 [3 4 5]
 [6 7 8]]
'''

# #### PANDAS###---------------------------------------动感光波--------------------------------------------
# numpy 和 pandas 不同点:
# pandas 更像字典形式的numpy
s = pd.Series([1, 3, 6, np.nan, 44, 1])  # 创建一个列表 np.nan =  none
print(s)
'''
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
dtype: float64
'''
dates = pd.date_range("20180716", periods=6)
print(dates)
'''
DatetimeIndex(['2018-07-16', '2018-07-17', '2018-07-18', '2018-07-19',
               '2018-07-20', '2018-07-21'],
              dtype='datetime64[ns]', freq='D')
'''
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)
'''
                   a         b         c         d
2018-07-16  1.363086  2.098875  1.898986  0.800066
2018-07-17  0.715738 -1.633167 -0.133430  0.951804
2018-07-18 -1.105968 -0.315523  1.216067  1.092425
2018-07-19 -1.549777  0.613988 -0.418058  0.808084
2018-07-20  0.959733  1.904160 -0.373022 -0.102294
2018-07-21  1.616347 -0.941505 -0.568996  0.863748

'''
df1 = pd.DataFrame(np.arange(12).reshape(3,4))
print(df1)

df2 = pd.DataFrame
df2 = pd.DataFrame({'A':1,
                    'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1,index =list(range(4)),dtype = 'float32'),
                    'D':np.array([3] * 4,dtype = 'int32'),
                    'E':pd.Categorical(['test','train','test','train']),
                    'F':'foo'})
print(df2)
print(df2.dtypes)
print (df2.index)
print(df2.columns)
print(df2.values)
print(df2.describe())
print(df2.T)
print(df2.sort_index(axis=1, ascending=False))
print(df2.sort_index(axis=0, ascending=False))
print (df2.sort_values(by='E', ascending=False))
