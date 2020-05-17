# coding: utf-8
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 看指定路径有没有我们要用的ckpt模型，没有就退出
savePath = '/Users/tal/Desktop/tf/ckpt2/'
saveFile = os.path.join(savePath, 'mnist_model.ckpt.meta')
if os.path.exists(saveFile) is False:
    print('Not found ckpt file!')
    exit()

# 我们要保存的pb模型的路径
savePbPath = os.path.join(savePath, 'model_pb')
if os.path.exists(savePbPath) is False:
    os.mkdir(savePbPath)
# 我们要保存的pb模型的文件名
savePbFile = os.path.join(savePbPath, 'mnist_model.pb')

with tf.Session() as sess:
    # 加载图
    saver = tf.train.import_meta_graph(saveFile)

    # 使用最后一次保存的
    saver.restore(sess, tf.train.latest_checkpoint(savePath))

    # 我们要固化哪些tensor
    output_graph_def = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        # output_node_names=['input_x', 'input_y', 'keep_prob', 'y_conv','predict']
        output_node_names=['output']
    )

    # 保存
    with tf.gfile.GFile(savePbFile, 'wb') as fd:
        fd.write(output_graph_def.SerializeToString())