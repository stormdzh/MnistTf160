# -*- coding:utf-8 -*-
import os

import tensorflow as tf

import mnist.input_data
import mnist.model as model

from tensorflow.python.framework import graph_util

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=True)

print ("训练数据共："+str(len(data.train.images)))
print ("测试数据共："+str(len(data.test.images)))
print ("第一个测试图的维度："+str(data.test.images[0].shape))
print ("第一个测试图的真实值："+str(data.test.labels[0]))

'''用线性模型训练数据并保存模型训练结果'''

# create model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
pre_num = tf.argmax(y, 1, output_type='int32',name="output")  # 输出节点名：output
correct_prediction = tf.equal(pre_num, tf.argmax(y_, 1 ,output_type='int32'))
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("variables：",variables)
saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print("测试数据：",sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))

    if not os.path.exists('data'):
        os.mkdir('data')

    # 保存模型到文件
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False)

    print("Saved:", path)

    savePbPath = '/Users/tal/PycharmProjects/MnistTf160/data/mnist_regression.pb'
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
    # with tf.gfile.FastGFile('model/mnist.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    #     f.write(output_graph_def.SerializeToString())
    with tf.gfile.GFile(savePbPath, 'wb') as fd:
        fd.write(output_graph_def.SerializeToString())