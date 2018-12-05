import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float64(np.random.rand(100, 2))  # 随机输入
y_data = np.dot(x_data, [0.100, 0.200]) + 0.3
y_data = y_data[:, None]

train_rate = 0.01
batch_size = 30
num_step = 2000
x_input = tf.placeholder(tf.float32, [None, 2])
y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1])

y1 = slim.fully_connected(x_input, 10)
y = slim.fully_connected(y1, 1)

# 最小化方差
# loss = tf.reduce_mean(tf.square(y - y_input))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=y, labels=y_input))
optimizer = tf.train.GradientDescentOptimizer(train_rate)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, num_step+1):
    # i = np.random.randint(60, size= (1))[0]
    batch_x, batch_y = train.next_batch(batch_size)
    sess.run(train,
             feed_dict={x_input: batch_x,
                        y_input: batch_y})
    if step % 20 == 0:

        print(sess.run(loss,
                       feed_dict={x_input: batch_x,
                                  y_input: batch_y}))

    #   得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

