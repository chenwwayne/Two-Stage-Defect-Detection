from __future__ import division
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

from model import ResNet
from utils import *
from dataset import ListDataset
from test import evaluate
import sys
import datetime
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=20, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/adc.data", help="path to data config file")
    # parser.add_argument("--pretrained_weights", type=str, default="config/yolov3_ckpt_5.pth")  # models/model1/yolov3_ckpt_73.pth
    parser.add_argument("--pretrained_weights", type=str)  # models/model1/yolov3_ckpt_73.pth
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=[896, 896], help="size of each image dimension")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--multiscale", default='False', choices=['True', 'False'])
    parser.add_argument("--augment", default='False', choices=['True', 'False'])
    parser.add_argument("--save_path", type=str, default='models/weights_1350_0102', help="save model path")
    parser.add_argument("--debug", type=str, default='False', choices=['True', 'False'], help="debug")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    args = parser.parse_args(argv)

    args.debug = True if args.debug == 'True' else False
    args.multiscale = True if args.multiscale == 'True' else False
    args.augment = True if args.augment == 'True' else False
    print_args(args)

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))

    if args.debug:
        print('debug...')
        # import shutil
        # if os.path.exists(args.save_path):
        #     shutil.rmtree(args.save_path)
        args.evaluation_interval = 1
        # debug模式下先删除save_path,并每间隔一轮验证一次

    # assert not os.path.exists(args.save_path)
    # os.makedirs(args.save_path)

    # adc.dat下有train和valid两个dat还有anchor.txt的路径
    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    if args.debug:
        valid_path = train_path
    anchors = get_anchors(data_config['anchors']).to('cuda')

    model = ResNet(anchors).to('cuda')

    if args.pretrained_weights:
        print('pretrained weights: ', args.pretrained_weights)
        model.load_pretrained_weights(args.pretrained_weights)

    dataset = ListDataset(train_path, img_size=args.img_size, augment=args.augment, multiscale=args.multiscale)
    eval = evaluate(path=valid_path, img_size=args.img_size, batch_size=args.batch_size, debug=args.debug)

    if args.debug:
        dataset.img_files = dataset.img_files[:10 * args.batch_size]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        collate_fn=dataset.collate_fn,
    )
    print('Number train sample: ', len(dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    # 这里优化器和学习率是不是要调节？


    print('\n### train ...')
    t0 = time.time()
    best_fitness = 0.0
    # start epoch
    for epoch in range(args.epochs):
        model.train()

        lr = max(1e-10, args.lr * (0.95 ** epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch_i, (imgs, targets, _) in enumerate(dataloader):
            imgs = Variable(imgs.to('cuda'))
            # 训练集有经过augment_sequential，而验证集没有
            # targets=([[0.0000, 0.7328, 0.2808, 0.0934, 0.0808],
            #         [1.0000, 0.5255, 0.5466, 0.0596, 0.1587],
            #         [1.0000, 0.5585, 0.8077, 0.0553, 0.2250],
            #         [3.0000, 0.4519, 0.4351, 0.1365, 0.2048]], device='cuda:0')
            targets = Variable(targets.to('cuda'), requires_grad=False)

            yolo_map, _ = model(imgs)
            #  yolo_map.shape : [4,]  其中每个yolo_map的格式如下： batch,featuremap_h,featuremap_w,anchor_num,(x,y,w,h,conf)
            loss, metrics = model.loss(yolo_map, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch_i + 1) % 100 == 0 or (batch_i + 1) == len(dataloader):
                time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
                lr = optimizer.param_groups[0]['lr']
                loss = metrics["loss"]
                xy = metrics["xy"]
                wh = metrics["wh"]
                conf = metrics["conf"]
                loss_str = 'loss: {:<8.2f}'.format(loss)
                loss_str += 'xy: {:<8.2f}'.format(xy)
                loss_str += 'wh: {:<8.2f}'.format(wh)
                loss_str += 'conf: {:<8.2f}'.format(conf)
                epoch_str = 'Epoch: {:4}({:4}/{:4})'.format(epoch, batch_i + 1, len(dataloader))
                print('[{}]{} {} lr:{}'.format(time_str, epoch_str, loss_str, lr))
        print()

        # start eval
        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            save_model_epoch = 'yolov3_ckpt_{}.pth'.format(epoch)
            print(save_model_epoch)
            example_save_path = args.save_path

            # for conf in [0.3, 0.4, 0.5, 0.6]:
            for conf in [0.01]:
                # eval return: precision, recall, AP, f1, ap_class
                p, r, ap, f1 = eval(model, iou_thres=0.5, conf_thres=conf, nms_thres=0.5, save_path=example_save_path, vis=False)
                example_save_path = None
                print('precision:{}\trecall:{}\tAP:{}\tF1:{}\t'.format(p, r, ap, f1))

            # 寻找最佳的权重并保存
            fi = fitness([p, r, ap, f1])  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi
                model.save_weights(os.path.join(args.save_path, save_model_epoch))
        # end eval
    # end epoch

    # 打印训练时间
    print('%g epochs completed in %.3f hours.\n' % (args.epochs, (time.time() - t0) / 3600))



if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])
