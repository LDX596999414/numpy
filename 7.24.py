import numpy as np
import tensorflow as tf
# 创建数据
x_date = np.random.rand(100).astype(np.float32)
y_date = x_date*0.2+0.3
# 搭建模型
Weight = tf.Variable(tf.random_uniform(([1]), -0.1, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weight*x_date+biases
# 计算误差
loss = tf.reduce_mean(tf.square(y-y_date))
# 传播误差 梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# 初始化
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
# 训练
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(Weight), sess.run(biases))


