import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder = '/home/ldx/PycharmProjects/7.26'
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 从截断的正态分布中输出随机值。
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # 解决过拟合的有效手段
x_image = tf.reshape(xs, [-1, 28, 28, 1])


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)  # 28*28*32
h_pool1 = max_pool_2x2(h_conv1)   # 14*14*32

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)  # 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # 7*7*64

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])   # -1表示先不考虑输入图片例子维度, 将上一个输出结果展平

# layer_1 = tf.layers.dense(h_pool2_flat, 7*7*64, activation=tf.nn.relu)
# layer_2 = tf.layers.dense(layer_1, 7*7*64, activation=tf.nn.relu)
# out_layer = tf.layers.dense(layer_2, 10)
#
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#      logits=out_layer, labels=ys))
# prediction = loss_op
# optimizer = tf.train.AdamOptimizer(0.001)
# train_op = optimizer.minimize(loss_op)
#
# correct_pred = tf.equal(tf.argmax(out_layer, 1), tf.argmax(ys, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# init = tf.global_variables_initializer()

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 们考虑过拟合问题，可以加一个dropout的处理

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)



init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)