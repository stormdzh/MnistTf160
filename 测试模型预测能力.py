import os, sys
from tensorflow.python.platform import gfile
import tensorflow as tf


def get_all_layernames(pb_file_path):
    # get all layers name

    print(pb_file_path)
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
            print(tensor_name)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        get_all_layernames(sys.argv[1])
    get_all_layernames('/Users/tal/Desktop/tf/ckpt2/model_pb/mnist_model.pb')
    # get_all_layernames('/Users/tal/Desktop/tf/ckpt2/model_pb/mnist-tf1.0.1.pb')