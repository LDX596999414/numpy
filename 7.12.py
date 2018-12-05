import numpy as np

a = np.arange(9).reshape(3, 3)
print(a)
print(a[:, ::-1])
print(a[::-1, :])
print(a.T)
print(a@a.T)
print(a> 3)
print(a[a>3])
b = a[a>3]
print(b[::-1])
a[a<7] = 100
a[2,1] = 100
a[2,2] = 100
print(a)

c = np.arange(9).reshape(3,3)
print(np.where(4<c,1,0))
print(np.repeat(c,2,axis=1))

#   print(np.repeat(c,[1,2,],axis=1))

print(np.repeat(3,9).reshape(3,3))
help(np.repeat)
############################
d = np.arange(9).reshape(3, 3)
'''
[[0 1 2]   #0维    d[strat:stop:steps] => [strat stop) ++steps
 [3 4 5]   1
 [6 7 8]]  2
'''

print(d[:, :1:])  #包左不包右
print(d[1::, :1:])
'''
[[3]
 [6]]
'''
height = float(input("请输入你的身高："))
weight = float(input("请输入您的体重："))
bmi = weight/(height**2)
print("您的BMI的指数是："+str(bmi))
if bmi<18.5:
    print("您的体重过轻")
if bmi>=18.5 and bmi<24.9:
    print("您的体重在正常范围内！")
if bmi>=24.9:
    print("您的体重过重")
if bmi>=29.9:
    print("肥胖")










