import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
MNIST_date_folder = '/home/ldx/PycharmProjects/7.26'
mnist = input_data.read_data_sets(MNIST_date_folder, one_hot=True)
# one_hot=True 即独热码，作用是将状态值编码成状态向量，例如，数字状态共有0~9这10种，对于数字7，
# 将它进行one_hot编码后为[0 0 0 0 0 0 0 1 0 0]，这样使得状态对于计算机来说更加明确，
# 对于矩阵操作也更加高效。
learning_rate = 0.001
num_steps = 20000
batch_size = 128
display_step = 100

n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


def neural_net(x):
    layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu)
    out_layer = tf.layers.dense(layer_2, 10)
    return out_layer


logits = neural_net(X)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# 评估模型
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 这个函数的返回值并不是一个数，而是一个向量，如果要求交叉熵，我们要再做一步tf.reduce_sum操作,
# 就是对向量里面所有元素求和，最后才得到，如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))







