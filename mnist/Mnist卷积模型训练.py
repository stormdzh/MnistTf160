import os

import tensorflow as tf

import mnist.model as model
import mnist.input_data

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=True)

from tensorflow.python.framework import graph_util
# 定义模型
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    print("keep_prob:",keep_prob)
    y, variables = model.convolutional(x, keep_prob)

# train
y_ = tf.placeholder(tf.float32, [None, 10], name='y')
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
pre_num = tf.argmax(y, 1, output_type='int32',name="output")  # 输出节点名：output
correct_prediction = tf.equal(pre_num, tf.argmax(y_, 1 ,output_type='int32'))
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("a->",tf.cast(accuracy, tf.float32))

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/tmp/mnist_log/1', sess.graph)
    summary_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(2000):   # 训练次数
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    if not os.path.exists('data'):
        os.mkdir('data')

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'convalutional.ckpt'),
        write_meta_graph=False, write_state=False)

    print("Saved:", path)

    savePbPath = '/Users/tal/PycharmProjects/MnistTf160/data/mnist_convolutional.pb'
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
    # with tf.gfile.FastGFile('model/mnist.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    #     f.write(output_graph_def.SerializeToString())
    with tf.gfile.GFile(savePbPath, 'wb') as fd:
        fd.write(output_graph_def.SerializeToString())