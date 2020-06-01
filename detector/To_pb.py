import os
import torch
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict
from model import ResNet
from torch.autograd import Variable
from tensorflow.python.tools import freeze_graph


def Bottleneck(x, place, stride=1, expansion=2, sc=''):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        y = slim.conv2d(x, place, (1, 1), stride=1, scope=sc + '.b0.layer')
        y = slim.conv2d(y, place, (5, 5), stride=stride, scope=sc + '.b1.layer')
        y = slim.conv2d(y, place * expansion, (1, 1), stride=1, scope=sc + '.b2.layer',
                        activation_fn=tf.identity)
        if stride > 1 or x.get_shape()[3] != place * expansion:
            x = slim.conv2d(x, place * expansion, (3, 3), stride=stride, scope=sc + '.downsample.layer',
                            activation_fn=tf.identity)
        y = tf.nn.relu(x + y, name=sc + '.downsample.layer')
        return y


def make_layer(x, places, block, stride, expansion=2, sc=''):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        x = Bottleneck(x, places, stride=stride, expansion=expansion, sc=sc + '.0')
        for i in range(1, block):
            x = Bottleneck(x, places, stride=1, expansion=expansion, sc=sc + '.{}'.format(i))
        return x


def Upsample(x, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    x = tf.image.resize_nearest_neighbor(x, (new_height, new_width))
    x = tf.identity(x, name='upsampled')
    return x


def YOLOLayer(x, anchors, sc=''):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.identity, padding='SAME'):
        num_anchors = anchors.get_shape()[0]
        y = slim.conv2d(x, num_anchors * 5, (3, 3), stride=1, scope=sc + '.conv.layer')
        grid_size = y.shape.as_list()[1:3]
        feature_map = tf.reshape(y, [-1, grid_size[0], grid_size[1], num_anchors, 5])
        box_centers, box_sizes, conf_logits = tf.split(feature_map, [2, 2, 1], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)
        conf_logits = tf.nn.sigmoid(conf_logits)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers / [grid_size[1], grid_size[0]]

        box_sizes = tf.exp(box_sizes) * anchors  # anchors -> [w, h]
        boxes = tf.concat([box_centers, box_sizes, conf_logits], axis=-1)
        boxes = tf.reshape(boxes, (-1, grid_size[0] * grid_size[1] * num_anchors, 5))

        return boxes


model_path = '/home-ex/tclhk/chenww/t2/models/yolo_v3_x/d10/0410_data0409/yolov3_ckpt_105.pth'
out_model_root = '/home-ex/tclhk/chenww/t2/run/models/d10/0409/ '
# data_config = 'config/adc.data'
# data_config = parse_data_config(data_config)
# anchors_file = data_config['anchors']
# anchors = get_anchors(anchors_file)
batch_size = 1

model_dict = torch.load(model_path)
anchors = model_dict['anchors'].cpu()
state_dict = model_dict['net']
keys = [s for s in state_dict.keys()]
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
        if weight.ndim == 4:
            weight = np.transpose(weight, (2, 3, 1, 0))
        else:
            weight = np.transpose(weight, (1, 0))
        bias = state_dict[keys[i + 1]].cpu().numpy()
        nclass = bias.shape[0]
        i += 2
    print(key, weight.shape, bias.shape)
    params[key[:-7] + '/weights:0'] = weight
    params[key[:-7] + '/biases:0'] = bias

print(params.keys())
model = ResNet(anchors, Istrain=False)
model.load_state_dict(state_dict)
model.eval()

input_data = np.random.random((batch_size, 3, 768, 1024)).astype(np.float32)
with torch.no_grad():
    inputs = torch.from_numpy(input_data.copy())
    inputs = Variable(inputs, requires_grad=False)

    with torch.no_grad():
        _, outputs = model(inputs)
        y1 = outputs
#         outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.5)
#         outputs_numpy = []
#         for output in outputs:
#             if output is None:
#                 outputs_numpy.append(None)
#             else:
#                 outputs_numpy.append(output.detach().cpu().numpy())
# print(outputs_numpy)

#
with tf.device('/cpu'):
    input = tf.placeholder(tf.float32, [batch_size, 768, 1024, 3], name='input')
    # conf_thres = tf.placeholder(tf.float32, name='conf_thres')
    # nms_thres = tf.placeholder(tf.float32, name='nms_thres')

    tf_anchors = tf.convert_to_tensor(anchors)
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        x = slim.conv2d(input, stride=2, num_outputs=16, kernel_size=(7, 7), scope='conv1.layer')
        blocks = [3, 4, 5, 5]
        x1 = make_layer(x, 16, blocks[0], 2, sc='layer1')
        x2 = make_layer(x1, 32, blocks[1], 2, sc='layer2')
        x3 = make_layer(x2, 64, blocks[2], 2, sc='layer3')
        x4 = make_layer(x3, 128, blocks[3], 2, sc='layer4')

        x = Bottleneck(x4, 128, 1, expansion=1, sc='output4')
        boxes4 = YOLOLayer(x, tf_anchors[3], sc='yolo4')

        upsampe_size = x3.get_shape().as_list()
        x = Upsample(x, upsampe_size)
        x = tf.concat([x, x3], axis=-1)
        x = Bottleneck(x, 64, expansion=1, sc='output3')
        boxes3 = YOLOLayer(x, tf_anchors[2], sc='yolo3')

        upsampe_size = x2.get_shape().as_list()
        x = Upsample(x, upsampe_size)
        x = tf.concat([x, x2], axis=-1)
        x = Bottleneck(x, 32, expansion=1, sc='output2')
        boxes2 = YOLOLayer(x, tf_anchors[1], sc='yolo2')

        upsampe_size = x1.get_shape().as_list()
        x = Upsample(x, upsampe_size)
        x = tf.concat([x, x1], axis=-1)
        x = Bottleneck(x, 16, expansion=1, sc='output1')
        boxes1 = YOLOLayer(x, tf_anchors[0], sc='yolo1')

        boxes = tf.concat([boxes1, boxes2, boxes3, boxes4], axis=1)
        output = tf.identity(boxes, name='output')

    save_params = tf.trainable_variables()
    saver = tf.train.Saver(save_params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    update = []
    with sess.as_default():
        for v in save_params:
            name = v.name
            data = params[name]
            data = tf.convert_to_tensor(data)
            update.append(tf.assign(v, data))
        sess.run(update)

    path = 'tmp'
    if not os.path.exists(path): os.mkdir(path)
    pb = 'tmp/tmp.pb'
    pb_model = out_model_root + 'detection.pb'
    saver.save(sess, 'tmp/model', write_meta_graph=False)
    graph_def = tf.get_default_graph().as_graph_def()
    with tf.gfile.GFile(pb, 'wb') as f:
        f.write(graph_def.SerializeToString())
    freeze_graph.freeze_graph(input_graph=pb,
                              input_saver='',
                              input_binary=True,
                              input_checkpoint='tmp/model',
                              output_node_names='output',
                              restore_op_name='save/restore_all',
                              filename_tensor_name='save/Const:0',
                              output_graph=pb_model,
                              clear_devices=True,
                              initializer_nodes='')
    shutil.rmtree(path)

    y2 = sess.run(boxes, feed_dict={input: input_data.copy().transpose(0, 2, 3, 1)})

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
            y3 = sess.run(output, feed_dict={input: input_data.copy().transpose(0, 2, 3, 1)})

    for ii in range(batch_size):
        _y1 = y1[0].detach().cpu().numpy()
        _y2 = y2[0]
        _y3 = y3[0]

        print(_y1.reshape(-1), '\n')
        print(_y2.reshape(-1), '\n')
        print(_y3.reshape(-1), '\n')

        print(np.max(np.abs(_y1 - _y2)) < 4e-5)
        print(np.max(np.abs(_y1 - _y3)) < 4e-5)

        print(np.max(np.abs(_y1 - _y2)))
        print(np.max(np.abs(_y1 - _y3)))
