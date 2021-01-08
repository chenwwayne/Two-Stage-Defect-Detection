from __future__ import division
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

from model import ResNet
from utils import get_anchors, parse_data_config
from utils import non_max_suppression

import cv2
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from collections import OrderedDict,defaultdict
import argparse
import sys


def print_args(args):
    print()
    for key, value in vars(args).items():
        if value is None:
            value = 'None'
        print('{:<50}{}'.format(key, value))
    print('\n')

class predict():
    def __init__(self, weights_path, conf_thres=0.7, nms_thres=0.5):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        model_dict = torch.load(weights_path)

        anchors = model_dict['anchors'].to('cuda')

        self.model = ResNet(anchors, Istrain=False).to('cuda')
        self.model.load_state_dict(model_dict['net'])
        self.model.eval()

    def __call__(self, inputs):
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs, requires_grad=False).to('cuda')
        with torch.no_grad():
            _, outputs = self.model(inputs)
            outputs = non_max_suppression(outputs, conf_thres=self.conf_thres, nms_thres=self.nms_thres)
            outputs_numpy = []
            for output in outputs:
                if output is None:
                    outputs_numpy.append(None)
                else:
                    outputs_numpy.append(output.detach().cpu().numpy())
        return outputs_numpy


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# def bbox_iou(box1, box2, IsMin=False):
#     b1_x1, b1_y1, b1_x2, b1_y2 = box1
#     b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
#
#     inter_rect_x1 = np.maximum(b1_x1, b2_x1)
#     inter_rect_y1 = np.maximum(b1_y1, b2_y1)
#     inter_rect_x2 = np.minimum(b1_x2, b2_x2)
#     inter_rect_y2 = np.minimum(b1_y2, b2_y2)
#     inter_area = (inter_rect_x2 - inter_rect_x1) * (inter_rect_y2 - inter_rect_y1)
#     b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#     b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
#
#     if IsMin:
#         iou = inter_area / (np.minimum(b1_area, b2_area) + 1e-16)
#     else:
#         iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
#
#     return iou

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

#compute_overlap(gt_boxes, pre_boxes)
def compute_overlap(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    # 行数为pre_bbox的数量
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        # print("box2", box2)
        # print("boxes1", boxes1)
        # 算1个gt_box和多个pre_box的iou，写入对应overlaps对应列
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
        # print(box2,boxes1)
        # print("overlaps[:, i]",overlaps[:, i])
    return overlaps



def draw_detect_gt_box(img, preboxes, gtboxes):
    h, w, _ = img.shape
    if preboxes is not None:
        for pre_box in preboxes:
            x1, y1, x2, y2, conf = pre_box
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            xx = min(x1, w - 80)
            yy = y1 - 5 if y1 > 30 else y2 + 25
            if y1 < 30 and y2 > h - 25: yy = h // 2
            cv2.putText(img, '{:.2f}'.format(conf), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, \
                        0.85, (0, 0, 255), 1)
    if gtboxes is not None:
        for gt_box in gtboxes:
            x1, y1, x2, y2 = gt_box
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return img


def get_result(pretrained_weights, test_data_list, output_txt_path, image_example, conf_thres=0.7, iou_thres=0.3):
    pre = predict(pretrained_weights, conf_thres=conf_thres)
    # test_data_path = os.path.realpath(test_data_path)

    miss_shot = os.path.join(image_example, 'FP_samples')
    if not os.path.exists(miss_shot):
        os.mkdir(miss_shot)

    with open(test_data_list) as f:
        filenames = [s.strip().split() for s in f.readlines()]
    print('Number files: ', len(filenames))

    class_list = list(set([fln[1] for fln in filenames]))
    print(class_list)

    cla_to_id = OrderedDict()
    for idx, name in enumerate(class_list):
        cla_to_id[name] = idx
    print(cla_to_id)

    mat = np.zeros((len(class_list), 3), dtype=np.int32)
    TSFAS_gt_num=0
    TSFAS_TN_num=0
    TSFAS_FP_num=defaultdict(int)
    # 需要存储每一类的TP，FP，TN
    # TSFAS当不给出框时认为是TP，每给一个框就是一个FP，TN默认为1
    # 以下的代码还需添加过滤FAS类的功能
    with open(output_txt_path + 'pred_output.dat', 'w') as f:
        for i, str in enumerate(tqdm(filenames)):
            filename, c = str[:2]
            img = cv2.imread(filename)
            proces_img = cv2.resize(img, (1024, 768))
            h, w, _ = img.shape

            inputs = proces_img.transpose(2, 0, 1).astype(np.float32) / 255.0
            inputs = np.expand_dims(inputs, 0)
            outputs = pre(inputs)
            assert (len(outputs) == 1)
            output = outputs[0]

            miss_shot_class_path = os.path.join(miss_shot, c)
            if not os.path.exists(miss_shot_class_path):
                os.mkdir(miss_shot_class_path)

            if c =='TSFAS':  # GT为TSFAS
                TSFAS_gt_num += 1
                if output is None:  # GT为TSFAS，推理结果无框
                    TSFAS_TN_num += 1
            elif output is None:  # 其他有缺陷类，却输出无框
                TSFAS_FP_num[c] += 1

            gt_boxes = np.asarray(str[2:], dtype=np.float32).reshape(-1, 5)[:, 1:5]
            # print('gt_boxes_xywh',gt_boxes)
            gt_boxes = xywh2xyxy(gt_boxes)
            # print('gt_boxes_xyxy',gt_boxes)
            f.write('{} {}'.format(filename, c))

            if output is not None:
                pre_boxes = np.asarray([box[0:4] for box in output])
                # print('pre_box_xyxy',pre_boxes)
                # assert 1==7
                overlap = compute_overlap(gt_boxes, pre_boxes)
                mat[cla_to_id[c], 1] += len(pre_boxes)
                mat[cla_to_id[c], 2] += len(gt_boxes)

                num_miss = 0
                for ovp in overlap:
                    if np.max(ovp) > iou_thres:
                        # gt和至少一个pred. box命中
                        mat[cla_to_id[c], 0] += 1
                    else:
                        num_miss += 1

                # 须要全部命中 GT_BOX ,num_miss才会 = 0
                if num_miss > 0 or len(overlap) == 0:  # 缺陷类预测的框与GT框 < iou_th  or  iou == 0
                    # print(overlap)
                    img = draw_detect_gt_box(img, output, gt_boxes)
                    cv2.imwrite(os.path.join(miss_shot_class_path, os.path.basename(filename)), img)

                for op in output:
                    x = (op[0] + op[2]) / 2
                    y = (op[1] + op[3]) / 2
                    w = (op[2] - op[0])
                    h = (op[3] - op[1])
                    f.write(' {} {} {} {} {}'.format(op[-1], x, y, w, h))
            else:
                mat[cla_to_id[c], 2] += len(gt_boxes)
                if c == 'TSFAS':
                    f.write('\n')
                    continue
                img = draw_detect_gt_box(img, output, gt_boxes)
                cv2.imwrite(os.path.join(miss_shot_class_path, os.path.basename(filename)), img)
            f.write('\n')

    # mat[cla_to_id['TSFAS'], :] = 0
    # print(mat)
    f.close()
    # mat[cla_to_id['TSFAS'],:]=0
    print('class  TP     PR     GT     precition recall')

    for name, dat in zip(class_list, mat):
        if name == 'TSFAS':
            continue
        prstr = ''
        prstr += '{:<7}'.format(name)
        prstr += '{:<7}{:<7}{:<7}'.format(*dat)
        prstr += '{:.4f}   {:.4f}'.format(dat[0] / dat[1], dat[0] / dat[2])
        print(prstr)
    print('total precition {}'.format(np.sum(mat[:, 0]) / np.sum(mat[:, 1])))
    print('total recall {}'.format(np.sum(mat[:, 0]) / np.sum(mat[:, 2])))
    # print('TSFAS recall {}'.format(TSFAS_TN_num/TSFAS_gt_num))
    # print('TSFAS confusion dict {}'.format(TSFAS_FP_num))

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--test_data_path", type=str, default="/home-ex/tclhk/chenww/t2/yolo_v3_x/config/2253_from_x/val.dat")
    parser.add_argument("--output_txt", type=str, default='/home-ex/tclhk/chenww/t2/models/yolo_v3_x/2253_weights/weights_1227/weights_1227_test/ep187_predict_result/D13-1350_bbox.dat')
    parser.add_argument("--image_example", type=str)
    parser.add_argument("--conf_thres", type=float, default=0.3)
    parser.add_argument("--iou_thres", type=float, default=0.0)
    args = parser.parse_args(argv)

    # args.debug = True if args.debug == 'True' else False
    # args.multiscale = True if args.multiscale == 'True' else False
    # args.augment = True if args.augment == 'True' else False
    print_args(args)

    if not os.path.exists(args.image_example):
        os.makedirs(args.image_example)

    get_result(args.pretrained_weights, args.test_data_path, args.output_txt, args.image_example, conf_thres=args.conf_thres,
               iou_thres=args.iou_thres)


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])