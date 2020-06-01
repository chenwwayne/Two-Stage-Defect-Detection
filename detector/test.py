
from __future__ import division

from model import ResNet
from dataset import ListDataset
from utils import get_anchors, print_args, parse_data_config
from utils import xywh2xyxy, non_max_suppression, get_batch_statistics

import os
import cv2
import argparse
import torch
import numpy as np
from torch.autograd import Variable

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

class evaluate():
    def __init__(self, path, img_size, batch_size, debug=False):
        dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
        if debug:
            dataset.img_files = dataset.img_files[:batch_size]
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
        )
        
    def __call__(self, model, iou_thres, conf_thres, nms_thres, save_path=None):
        # iou_thres = 0.5,
        # conf_thres = 0.7, 
        # nms_thres = 0.5,
        # img_size = args.img_size,
        # batch_size = 8,
        model.eval()

        if save_path is not None:
            example = os.path.join(save_path, 'example')
            if not os.path.exists(example): os.mkdir(example)
            file_idx = 0
        sample_metrics = None
        for batch_i, (inputs, targets, cs) in enumerate(self.dataloader):
            imgs = inputs.detach().cpu().numpy().copy() * 255
            if targets.size(0):
                targets[:, 1:] = xywh2xyxy(targets[:, 1:])

            inputs = Variable(inputs, requires_grad=False).to('cuda')

            with torch.no_grad():
                _, outputs = model(inputs)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            """
            # 检测可视化
            if save_path is not None:
                for idx, img in enumerate(imgs):
                    img = img.astype(np.uint8).transpose(1, 2, 0).copy()
                    c, filename = cs[idx]
                    filename = os.path.basename(filename)
                    cv2.putText(img, os.path.join(c, filename), (50, 30), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    h, w, _ = img.shape
                    sacle = np.asarray([w, h, w, h], dtype=np.float32)

                    # 画GT-框
                    true_box = 0
                    if targets.size(0):
                        target = targets[targets[:, 0] == idx][:, 1:]
                        for bbox in target.clone():
                            true_box += 1
                            x1, y1, x2, y2 = (bbox.numpy() * sacle).astype(np.int32)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    #  画预测框
                    output = outputs[idx]
                    nbox = 0
                    if output is not None:
                        output = output.detach().cpu()
                        nbox = output.size(0)
                        for bbox in output.clone():
                            conf = bbox[4]
                            bbox = bbox[:4]
                            x1, y1, x2, y2 = (bbox.numpy() * sacle).astype(np.int32)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            xx = min(x1, w - 80)
                            yy = y1 - 5 if y1 > 30 else y2 + 25
                            if y1 < 30 and y2 > h - 25:
                                yy = h // 2
                            cv2.putText(img, '{:.2f}'.format(conf), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, \
                                        0.85, (0, 0, 255), 2)

                    save_img_path = os.path.join(example, c)
                    if not os.path.exists(save_img_path):
                        os.mkdir(save_img_path)
                    cv2.imwrite(os.path.join(save_img_path, filename), img)
                    file_idx += 1
            """
            if sample_metrics is None:
                sample_metrics = get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
            else:
                sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        image_acc = sample_metrics[:, 4] / (sample_metrics[:, 0] + 1e-16)
        bbox_acc = sample_metrics[1, 3] / (sample_metrics[1, 2] + 1e-16)
        bbox_rec = sample_metrics[1, 3] / (sample_metrics[1, 1] + 1e-16)
        # names = ['image', 'ture', 'det', 'box_acc', 'image_acc']
        return sample_metrics, image_acc[0], image_acc[1], bbox_acc, bbox_rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/v4/adc.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="/home-ex/tclhk/chenww/t2/models/yolo_v3_x/0117_v4/yolov3_ckpt_34.pth",
                        help="path to weights file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")  # 计算指标用，比方说iou>0.5才算召回
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.0, help="iou thresshold for non-maximum suppression")  # nms函数用，两个bbox若iou大于nms_th，则滤除
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=[768, 1024], help="size of each image dimension")
    args = parser.parse_args()
    print_args(args)

    data_config = parse_data_config(args.data_config)
    valid_path = data_config['valid']
    nclass = int(data_config['classes'])
    anchors = get_anchors(data_config['anchors']).to('cuda')

    model = ResNet(anchors).to('cuda')
    model.load_state_dict(torch.load(args.weights_path)['net'])

    print('Compute mAP...')
    save_path = '/home-ex/tclhk/chenww/t2/models/yolo_v3_x/0117_v4/test_result_ep34_iou0.5_conf0.7_nms0.0/'
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    eval = evaluate(path=valid_path, img_size=args.img_size, batch_size=args.batch_size)

    # sample_metrics, image_acc0, image_acc1, bbox_acc, bbox_rec = eval(
    #     model,
    #     iou_thres=args.iou_thres,
    #     conf_thres=args.conf_thres,
    #     nms_thres=args.nms_thres,
    #     save_path=save_path,
    # )

    metrics = eval(
        model,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        save_path=save_path,
    )

    print('image_acc: {}\t{}\tbbox_acc: {}\tbbox_recall: {}'.format(*metrics[1:]))

    names = ['image', 'ture', 'det', 'box_acc', 'image_acc']
    print('{:<10}{:<10}{:<10}{:<10}{:<10}'.format(*names))
    print('{:<10}{:<10}{:<10}{:<10}{:<10}'.format(*metrics[0][0]))
    print('{:<10}{:<10}{:<10}{:<10}{:<10}'.format(*metrics[0][1]))

    # print("Average Precisions:")
    # print(f"image_acc: {image_acc0}\t{image_acc1}\tbbox_acc: {bbox_acc}\tbbox_recall: {bbox_rec}")
    # print('sample_metrics: ', str(sample_metrics.tolist()))

