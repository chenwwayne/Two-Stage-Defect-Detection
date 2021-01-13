from __future__ import division
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import cv2
from model import ResNet
import numpy as np
import shutil
import torch
from torch.autograd import Variable
from random import shuffle
from collections import OrderedDict
from dataset import crop

test_data = 'result/TEST_BATCH_3.dat'
test_files = []
with open(test_data) as ftxt:
    lines = ftxt.readlines()

true_labels = []
pred_labels = []
for line in lines:
    line = line.strip().split()
    filename = line[0]
    true_class = line[1]
    pred_class = line[2]
    conf = float(line[3])
    bbox = np.asarray(line[4:], dtype=np.float32).reshape(-1,5)
    true_labels.append(true_class)
    pred_labels.append(pred_class)
    test_files.append((filename, true_class, pred_class, conf, bbox))

true_class_names = list(set(true_labels))
pred_class_names = list(set(pred_labels))
nfiles = len(test_files)
print('test files:', nfiles)

counts1 = [true_labels.count(c) for c in pred_class_names]
others =  [c for c in true_class_names if c not in pred_class_names]
counts2 = [true_labels.count(c) for c in others]

Class2Id = OrderedDict()
Id2Class = OrderedDict()
idx = 0
for i in np.argsort(counts1)[::-1]:
    class_name = pred_class_names[i]
    Class2Id[class_name] = idx
    Id2Class[idx] = class_name
    idx += 1
    
for i in np.argsort(counts2)[::-1]:
    class_name = others[i]
    Class2Id[class_name] = idx
    Id2Class[idx] = class_name
    idx += 1
print(Class2Id)
print(Id2Class)

thresh = [0.1 * i for i in range(10)]
mat = np.zeros((len(thresh), len(true_class_names), len(pred_class_names)+1), dtype=np.int32)
result = np.zeros((len(thresh), 2), dtype=np.float32)
mat_display = []

for t, th in enumerate(thresh):
    # for i, y in enumerate(features):
    for img_file, true_class, pred_class, conf, bbox in test_files:
        c_label = Class2Id[true_class]
        p_label = Class2Id[pred_class]

        if conf < th:
            pass
            mat[t, c_label, -1] +=1
        else:
            mat[t, c_label, p_label] +=1

        mat_str = '    \t' + '\t'.join([key for key, _ in Class2Id.items() if key in pred_class_names]) + '\tNONO \n'

        # if t == 0:
            # result_path = os.path.join(result_dir, '{}_{}'.format(class_name, Id2Class[pred_label]))
            # if not os.path.exists(result_path):
                # os.mkdir(result_path)
            # img  = cv2.imread(img_file)
            # h, w, _ = img.shape
            # idx = np.argmax(bbox[:,0])
            # for ii, (conf, cx, cy, cw, ch) in enumerate(bbox):
                # x1 = int(w*(cx - cw/2))
                # y1 = int(h*(cy - ch/2))
                # x2 = int(cw*w) + x1
                # y2 = int(ch*h) + y1
                # color = (0,0,255) if ii == idx else (255,255,255)
                # cv2.rectangle(img, (x1, y1), (x2, y2), color)

            # y_start = 100
            # for idx in pred_idx[:5]:
                # put_txt = '{:}:{:>7.3f}'.format(Id2Class[idx], y[idx])
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255))
                # xx = min(x1, w - 80)
                # yy = y1 - 5 if y1 > 30 else y2 + 25
                # if y1 < 30 and y2 > h - 25: yy = h // 2
                # cv2.putText(img, '{:.2f}'.format(conf), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, \
                            # 0.85, color, 1)
            # filename = os.path.basename(img_file)
            # cv2.imwrite(os.path.join(result_path, filename), img)

    accs, counts = 0.0, 0.0
    for h_, h in enumerate(mat[t].tolist()):
        mat_str += Id2Class[h_] + '\t'
        acc = 0
        count = 0
        for w_, w in enumerate(h):
            count += w
            if h_ == w_ and w_ < len(pred_class_names):
                mat_str += '|{:<4}\t'.format(w)
                acc += w
            else:
                mat_str += ' {:<4} \t'.format(w)
        accs += acc
        counts += count
        mat_str += '\t{:<4}\t{:<4.2f}\n'.format(count, acc/(count+1e-16))
    # print(mat_str)
    mat_display.append(mat_str)
    coverage = mat[t, :, :-1].sum()+1e-16
    result[t, 0] = coverage / counts
    result[t, 1] = accs / coverage

for i, mat_str in enumerate(mat_display):
    print('thresh: {:.2f} recall: {:.2f} precision: {:.2f}'.format(thresh[i], *result[i]))
    print(mat_str)

for i, th in enumerate(thresh):
    print('thresh: {:.2f} recall: {:.2f} precision: {:.2f}'.format(th, *result[i]))

labels = None
print('\n\n')
