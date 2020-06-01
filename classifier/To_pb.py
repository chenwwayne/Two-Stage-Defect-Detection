import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict
from model import ResNet
from torch.autograd import Variable
from tensorflow.python.tools import freeze_graph

model_path = '/home-ex/tclhk/chenww/t2/models/classification_x/d10/0410_data0409/model_ckpt_62.pth'
out_model_root = '/home-ex/tclhk/chenww/t2/run/models/d10/0409/'
model_dict = torch.load(model_path)
class_name = model_dict['class_name']
state_dict = model_dict['net']

keys = [s for s in state_dict.keys() if s != '_centerloss.centers']
print(keys)

i = 0
eps = 1e-05
params = OrderedDict()
while i < len(keys):
    key = keys[i]
    weight = state_dict[key].cpu().numpy()
    if 'bn' in keys[i + 1]:
        w = state_dict[keys[i + 1]].cpu().numpy()
        b = state_dict[keys[i + 2]].cpu().numpy()
        m = state_dict[keys[i + 3]].cpu().numpy()
        s = state_dict[keys[i + 4]].cpu().numpy()
        assert state_dict[keys[i + 5]].size() == torch.Size([]), str(state_dict[keys[i + 5]].size())
        i += 6

        s = w / np.sqrt(s + eps)
        weight = weight * s[:, None, None, None]
        weight = np.transpose(weight, (2, 3, 1, 0))
        bias = b - m * s

    else:
        weight = np.transpose(weight, (1, 0))
        bias = state_dict[keys[i + 1]].cpu().numpy()
        nclass = bias.shape[0]
        i += 2
    print(key, weight.shape, bias.shape)
    params[key[:-7] + '/weights:0'] = weight
    params[key[:-7] + '/biases:0'] = bias

model = ResNet(class_name=class_name)
model.load_state_dict(state_dict)
model.eval()

input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
with torch.no_grad():
    inputs = torch.from_numpy(input_data.copy())
    inputs = Variable(inputs, requires_grad=False)
    f1, y1 = model(inputs)
    y1 = torch.sigmoid(y1).detach().numpy()
    f1 = f1.detach().numpy()


def Bottleneck(x, place, stride=1, downsampling=False, expansion=2, sc=''):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        y = slim.conv2d(x, place, (1, 1), stride=1, scope=sc + '.b0.layer')
        y = slim.conv2d(y, place, (5, 5), stride=stride, scope=sc + '.b1.layer')
        y = slim.conv2d(y, place * expansion, (1, 1), stride=1, scope=sc + '.b2.layer',
                        activation_fn=tf.identity)
        if downsampling:
            x = slim.conv2d(x, place * expansion, (3, 3), stride=stride, scope=sc + '.downsample.layer',
                            activation_fn=tf.identity)
        y = tf.nn.relu(x + y, name=sc + '.downsample.layer')
        return y

def make_layer(x, places, block, stride, expansion=2, sc=''):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        x = Bottleneck(x, places, stride=stride, downsampling=True, expansion=expansion, sc=sc + '.0')
        for i in range(1, block):
            x = Bottleneck(x, places, stride=1, downsampling=False, expansion=expansion, sc=sc + '.{}'.format(i))
        return x


with tf.device('/cpu'):
    input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
    tf_class_name = tf.convert_to_tensor(class_name, dtype=tf.string, name='class_name')
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        x = slim.conv2d(input, stride=2, num_outputs=32, kernel_size=(7, 7), scope='conv1.layer')

        blocks = [3, 4, 6, 3]
        x = make_layer(x, 32, blocks[0], 2, sc='layer1')
        x = make_layer(x, 64, blocks[1], 2, sc='layer2')
        x = make_layer(x, 128, blocks[2], 2, sc='layer3')
        x = make_layer(x, 256, blocks[3], 2, sc='layer4')

        x = slim.avg_pool2d(x, (7, 7), 1, 'VALID')
        x = slim.flatten(x)
        xx = x
        x = slim.fully_connected(x, nclass, activation_fn=tf.identity, scope='fc')
        x = tf.sigmoid(x)
        x = tf.identity(x, name='output')


    save_params = tf.trainable_variables()
    saver = tf.train.Saver(save_params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    update = []
    with sess.as_default():
        for v in save_params:
            name = v.name
            print(name)
            data = params[name]
            data = tf.convert_to_tensor(data)
            update.append(tf.assign(v, data))
        sess.run(update)

    path = 'tmp'
    if not os.path.exists(path):os.mkdir(path)
    pb = 'tmp/tmp.pb'
    pb_model=out_model_root + 'classify.pb'
    saver.save(sess, 'tmp/model', write_meta_graph=False)
    graph_def = tf.get_default_graph().as_graph_def()
    with tf.gfile.GFile(pb, 'wb') as f:
        f.write(graph_def.SerializeToString())
    freeze_graph.freeze_graph(input_graph=pb,
                              input_saver='',
                              input_binary=True,
                              input_checkpoint='tmp/model',
                              output_node_names='output,class_name',
                              restore_op_name='save/restore_all',
                              filename_tensor_name='save/Const:0',
                              output_graph=pb_model,
                              clear_devices=True,
                              initializer_nodes='')
    shutil.rmtree(path)

    f2, y2 = sess.run([xx, x], feed_dict={input: input_data.copy().transpose(0, 2, 3, 1)})

    tf.reset_default_graph()
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_model, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            output = sess.graph.get_tensor_by_name("output:0")
            input = sess.graph.get_tensor_by_name("input:0")
            class_name = sess.graph.get_tensor_by_name("class_name:0")
            y3, cc = sess.run([output,class_name], feed_dict={input: input_data.copy().transpose(0, 2, 3, 1)})
            cc= [str(c,'utf-8') for c in cc]
            print(cc)
            # print(type(cc), cc.dtype, cc.shape)
            exit()

    print(f1.shape, f2.shape)
    print(f1.reshape(-1), '\n')
    print(f2.reshape(-1), '\n')
    print(y1.reshape(-1), '\n')
    print(y2.reshape(-1), '\n')
    print(y3.reshape(-1), '\n')
    print(np.max(np.abs(f1 - f2)) < 1e-5)
    print(np.max(np.abs(y1 - y2)) < 1e-5)
    print(np.max(np.abs(y1 - y3)) < 1e-5)
    print(np.max(np.abs(f1 - f2)))
    print(np.max(np.abs(y1 - y2)))
    print(np.max(np.abs(y1 - y3)))
