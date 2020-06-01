import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import cv2
from model import ResNet
import numpy as np
import torch
from torch.autograd import Variable
from random import shuffle
from collections import OrderedDict
import shutil
import sys
import argparse

debug = True
crop_size = 224

def print_args(args):
    print()
    for key, value in vars(args).items():
        if value is None:
            value = 'None'
        print('{:<50}{}'.format(key, value))
    print('\n')

def get_result(pretrained_weights, test_data, result_txt_dir):
    
    with open(test_data) as ftxt:
        lines = ftxt.readlines()

    test_files = []
    for line in lines:
        line = line.strip().split()
        filename = line[0]
        classname = line[1]
        boxes = np.asarray(line[2:], dtype=np.float32).reshape(-1, 5)
        test_files.append((filename, classname, boxes))

    if debug:
        shuffle(test_files)
        test_files = test_files[:]

    model_dict = torch.load(pretrained_weights)
    class_name = model_dict['class_name']
    
    state_dict = model_dict['net']
    
    nclass = len(class_name)
    nfiles = len(test_files)
    print(class_name)
    print('     '.join(['{}:{}'.format(i, c) for i, c in enumerate(class_name)]))
    print('nclass:', nclass)
    print('test files:', nfiles)

    model = ResNet(class_name = class_name)
    model.to('cuda')

    model.load_state_dict(state_dict)
    model.eval()
    
    with open(result_txt_dir + 'pred_result.txt', 'w') as ftxt:
        for i, (image_file, classname, boxes) in enumerate(test_files):
            if (i+1) % 500 == 0 or (i+1) == nfiles:
                print(i+1, nfiles)
            img =cv2.imread(image_file)
            assert img is not None, image_file
            h, w, _ = img.shape
            if boxes.shape[0]:
                # print(boxes)
                boxes_str = ' '.join([' '.join(map(str, s)) for s in boxes])
                
                index = np.argmax(boxes[:, 0]*(boxes[:, 3] + boxes[:, 4]))
                _, cx, cy, _, _ = boxes[index]
                x1 = int(cx*w - crop_size/2)
                y1 = int(cy*h - crop_size/2)
                x1 = min(max(0, x1), w - crop_size)
                y1 = min(max(0, y1), h - crop_size)
                x2 = x1 + crop_size
                y2 = y1 + crop_size
                img_crop = img[y1:y2, x1:x2, :]
                # cv2.imwrite('{}.jpg'.format(i), img_crop)
                
                inputs_numpy = np.transpose(img_crop, (2, 0, 1))
                inputs_numpy = np.expand_dims(inputs_numpy.astype(np.float32), 0)

                with torch.no_grad():
                    inputs = torch.from_numpy(inputs_numpy/255)
                    inputs = Variable(inputs.to('cuda'), requires_grad=False)

                    f, y = model(inputs)
                    y = torch.sigmoid(y).detach().cpu().numpy()
                    index = np.argmax(y[0])
                    label = class_name[index]
                    conf = y[0, index]
                    ftxt.write('{} {} {} {} {}\n'.format(image_file, classname, label, conf, boxes_str))
            else:
                ftxt.write('{} {} TSFAS 1.0\n'.format(image_file, classname))

    

    print('\n\n')

# if __name__ == '__main__':
#     # pretrained_weights = './models/model21/model_ckpt_198.pth'
#     # result_txt = 'result/test.dat'
#     # test_data = './data/test.dat'
#     #
#     # pretrained_weights = './models/model22/model_ckpt_200.pth'
#     # result_txt = 'result/train.dat'
#     # test_data = './data/train.dat'
#
#     pretrained_weights = '/home-ex/tclhk/chenww/t2/models/classification_x/0103_v3/model_ckpt_30.pth'
#     result_txt = '/home-ex/tclhk/chenww/t2/models/classification_x/0103_v3/result_ep30/'
#     test_data = '/home-ex/tclhk/chenww/t2/classification_x/data/v3/val.dat'
#
#
#     print(pretrained_weights)
#     print(test_data)
#
#     get_result(pretrained_weights, test_data, result_txt)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--debug", type=int)
    args = parser.parse_args(argv)
    print_args(args)
    get_result(args.pretrained_weights, args.test_data, args.result_dir)


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1:])
