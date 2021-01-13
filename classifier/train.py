import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

import sys
import numpy as np
from dataset import DataLoader
from model import ResNet
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import datetime
import argparse

def print_args(args):
    print()
    for key, value in vars(args).items():
        if value is None:
            value = 'None'
        print('{:<50}{}'.format(key, value))
    print('\n')


def train(model, train_loader, valid_loader, args, debug=False):
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.pretrained_weights:
        print('pretrained weights: ', args.pretrained_weights)
        model.load_pretrained_weights(args.pretrained_weights)

    print('\n### train ...')
    for epoch in range(args.epochs):
        model.train()

        lr = max(1e-10, args.lr * (0.95 ** epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch_i in range(len(train_loader)):
            imgs, labels = train_loader.next()
            imgs = Variable(imgs.to('cuda'), requires_grad=True)
            labels = Variable(labels.to('cuda'))
            f, y = model(imgs)
            sce_loss, acc = model.loss(y, labels)
            center_loss = model.center_loss(f, labels, loss_weight=args.center_loss_factor)
            L2_loss = model.regularization_loss()

            loss = sce_loss + center_loss + L2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sce_loss = sce_loss.cpu().item()
            center_loss = center_loss.cpu().item()
            L2_loss = L2_loss.cpu().item()
            acc = acc.cpu().item()

            if (batch_i + 1) % 100 == 0 or (batch_i + 1) == len(train_loader) or debug:
                time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
                display = '[{}]'.format(time_str)
                display += 'Epoch: {:3}({:4}/{:4}) '.format(epoch, batch_i + 1, len(train_loader))
                display += 'loss: {:<10.3f}'.format(loss.cpu().item())
                display += 'sce: {:<10.3f}'.format(sce_loss)
                display += 'center_loss: {:<10.3f}'.format(center_loss)
                display += 'L2_loss: {:<10.3f}'.format(L2_loss)
                display += 'acc: {:<8.3f}'.format(acc)
                display += 'lr:{:.4e}'.format(lr)
                print(display)
        print()
        if epoch % args.evaluation_interval == 0 or debug:
            model.eval()
            result_all = []
            for valid_i in range(len(valid_loader)):
                with torch.no_grad():
                    imgs, labels = valid_loader.next()
                    inputs = Variable(imgs.to('cuda'), requires_grad=False)
                    labels = Variable(labels.to('cuda'), requires_grad=False)
                    f, y = model(inputs)
                    sce_loss, acc = model.loss(y, labels)
                    center_loss = model.center_loss(f, labels, loss_weight=args.center_loss_factor)
                    result_all.append([sce_loss.cpu().item(), center_loss.cpu().item(), acc.cpu().item()])

            save_model_epoch = 'model_ckpt_{}.pth'.format(epoch)
            model.save_weights(os.path.join(args.save_path, save_model_epoch))

            result_all = np.asarray(result_all).mean(0).tolist()

            print('Test cross_entropy: {}  center_loss: {} accuracy: {}'.format(*result_all))
            print('Save weights: ', save_model_epoch)
            print()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=150, help="size of each image batch")
    parser.add_argument("--pretrained_weights", type=str,help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--train_file", type=str, default='data/train.dat', help="image list file")
    parser.add_argument("--test_file", type=str, default='data/val.dat', help="image list file")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--save_path", type=str, default='models/weights_1350_0102', help="save model path")
    parser.add_argument("--debug", type=str, default='False', choices=['True', 'False'], help="debug")
    parser.add_argument("--augment", type=str, default='False', choices=['True', 'False'], help="augment")
    parser.add_argument("--random_crop", type=str, default='True', choices=['True', 'False'], help="debug")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--center_loss_factor', type=float, help='Center loss factor.', default=0)
    args = parser.parse_args(argv)
    print_args(args)

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    args.debug = True if args.debug == 'True' else False
    args.augment = True if args.augment == 'True' else False
    args.random_crop = True if args.random_crop == 'True' else False
    if args.debug:
        print('debug...')
        import shutil
        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path)
        args.evaluation_interval = 1

    # assert not os.path.exists(args.save_path)
    # os.makedirs(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    example = os.path.join(args.save_path, 'example')
    # os.mkdir(example)
    cudnn.benchmark = True

    train_loader = DataLoader(args.train_file, debug=args.debug, augment=args.augment, \
                              shuffle=True, batch_size=args.batch_size, \
                              num_workers=args.n_cpu, pin_memory=True, \
                              min_data_count =100, random_crop=args.random_crop,
                              )

    valid_loader = DataLoader(args.test_file, debug=args.debug, augment=False, \
                              shuffle=False, batch_size=args.batch_size, \
                              num_workers=args.n_cpu, \
                              class_list=train_loader.dataset.class_list
                              )
    # if not args.debug:
    #     train_example = os.path.join(example, 'train')
    #     valid_example = os.path.join(example, 'valid')
    #     train_loader.save_example(train_example)
    #     valid_loader.save_example(valid_example)

    train_loader.save_label_name(os.path.join(args.save_path, 'labels.name'))
    model = ResNet(class_name = train_loader.class_name)

    train(model, train_loader, valid_loader, args, debug=args.debug)


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])