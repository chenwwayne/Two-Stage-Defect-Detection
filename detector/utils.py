from __future__ import division
import torch
import numpy as np
import pdb


def print_args(args):
    print()
    for key, value in vars(args).items():
        if value is None:
            value = 'None'
        print('{:<50}{}'.format(key, value))
    print('\n')


def parse_data_config(path):
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        print(line)
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readlines()
        assert len(anchors) == 1
    anchors = np.asarray(anchors[0].strip().split(), dtype=np.float32)
    anchors = anchors.reshape(-1, 3, 2)

    return torch.from_numpy(anchors)


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    target_boxes = None
    # len(outputs) = batch_size
    # so, outputs[sample_i] 是一张图片对应的向量：[x,y,w,h,conf]
    for sample_i in range(len(outputs)):
        if targets.size(0):  # 存在标签
            __t = targets[targets[:, 0] == sample_i][:, 1:]
            if __t.size(0):
                target_boxes = __t

        if outputs[sample_i] is None:
            if target_boxes is None:#表明实际是TSFAS,预测也是TSFAS
                # check what can we do here
                pass
            else:
                # check what can we do here
                pass
            continue
        else: # 模型有输出结果
                # annotations = targets[targets[:, 0] == sample_i][:, 1:]
                # target_labels = annotations[:, 0] if len(annotations) else []
                # if len(annotations):
                #     detected_boxes = []
                #     target_boxes = annotations[:, 1:]

                    # for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                    #     if len(detected_boxes) == len(annotations):
                    #         break
                    #     if pred_label not in target_labels:
                    #         continue
                output = outputs[sample_i].detach().cpu()
                pred_boxes = output[:, :4]
                pred_scores = output[:, 4]
                # 因为我们只预测一个类别，所以将Pred_labels都置为 0, shape与pred_scores相同
                pred_labels = torch.full(pred_scores.shape, 0)

                true_positives = np.zeros(pred_boxes.shape[0])
                if target_boxes is not None:
                    detected_boxes = []
                    for pred_i, pred_box in enumerate(pred_boxes):
                        iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                        if iou >= iou_threshold and box_index not in detected_boxes:
                            true_positives[pred_i] = 1
                            detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

# output无缺陷图片的评估信息与缺陷的评估信息
def get_batch_statistics_ori(outputs, targets, iou_threshold):
    # outputs一个batch的输出，targets一个batch的标签
    # 两者从格式上来讲应该一致，[batch_size,n,5]
    #  0     1      2      3      4
    # FAS  ALL_GT                TN
    # OTS  ALL_GT  TP+FP   TP    
    batch_metrics = np.zeros((2, 5), dtype=np.int32)

    for sample_i, output in enumerate(outputs):
        # sample_i为实际采样得到的batch_id
        target_boxes = None
        image_idx = 0
        if targets.size(0):
            __t = targets[targets[:, 0] == sample_i][:, 1:]
            if __t.size(0):
                target_boxes = __t
                image_idx = 1
                batch_metrics[1, 1] += __t.size(0)
                # batch_i不为空的所有bbox
                # 1,1是实际有的gt数目

        # 如果没有targets，说明这个图片是背景类，image_idx=0
        batch_metrics[image_idx, 0] += 1

        if output is None:
            if target_boxes is None:
                # 表明是FAS类
                batch_metrics[image_idx, 4] += 1
        else:
            output = output.detach().cpu()
            pred_boxes = output[:, :4]
            batch_metrics[image_idx, 2] += pred_boxes.size(0)
            # 1，2prebox

            if target_boxes is not None:
                detected_boxes = []
                for pred_i, pred_box in enumerate(pred_boxes):
                    print(pred_box)
                    iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)

                    if iou >= iou_threshold and box_index not in detected_boxes:
                        detected_boxes += [box_index]
                        batch_metrics[image_idx, 3] += 1
                        # 1，3是TP数目
                        if pred_i == 0:
                            batch_metrics[image_idx, 4] += 1

    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True, IsMin=True):
    # print(x1y1x2y2)
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    if IsMin:
        iou = inter_area / (torch.min(b1_area, b2_area) + 1e-16)
    else:
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    xyxy = xywh2xyxy(prediction[..., :4])
    prediction[..., :4] = torch.clamp(xyxy, 0, 1)

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        if not image_pred.size(0):
            continue

        score = image_pred[:, 4]
        detections = image_pred[(-score).argsort()]

        keep_boxes = []
        while detections.size(0): # 检出框的个数
            #  unsqueeze在第arg维增加一个维度值为1的维度
            #  计算当前box与所有box的iou，得到向量[0,1,1,1,0,1...], iou > nms_th 则表明这个框要去除，用1表示
            invalid = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4], IsMin=False) > nms_thres
            keep_boxes += [detections[0]]
            detections = detections[~invalid] # 留下iou < nms_th 的框
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


#  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
#  faster r-cnn中 VOC-AP 的计算方法
def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))  # [0.  0.0666, 0.1333, 0.1333, 0.4, 0.4666,  1.]
    mpre = np.concatenate(([0.], prec, [0.]))  # [0.  0.0666, 0.1333, 0.1333, 0.4, 0.4666  0.]

    # compute the precision envelope
    # 计算出precision的各个断点(折线点)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # [1.     1.     0.6666 0.4285 0.3043 0.    ]

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]  # precision前后两个值不一样的点
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):
    # tp为one-hot向量，当predbox与gtbox的iou符合条件时为1
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    ap, p, r = [], [], []

    # 遍历所有类别
    for c in unique_classes:
        i = pred_cls == c # i为一个one-hot列表，当且仅当预测类别 == 当前遍历的其中一个类别时为1
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            # tp[i]为预测正确的所有类别中为当前类别A的onehot选中
            # (1 - tp[i])则是tp[i]的取反
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.1, 0.1, 0.8, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    x = np.array(x)
    return (x * w).sum()


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 480, 640),), verbose=False)
        fs = ', %.1f GFLOPS' % (macs / 1E9 * 2)
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))