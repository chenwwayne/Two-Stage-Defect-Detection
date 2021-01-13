import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import cv2
import json
from model import ResNet
import numpy as np
import torch
from torch.autograd import Variable
from random import shuffle
from collections import OrderedDict, defaultdict
import shutil
from tqdm import tqdm

val_dat = '/home-ex/tclhk/xie/ADC-code/classification/data/val.dat'
# confusion_img_path = None
confusion_img_path = '/home-ex/tclhk/xie/ADC-code/classification/data/vis/'
# model_weights = '/home-ex/tclhk/xie/ADC-code/classification/models/test1226/model_ckpt_166.pth'
model_weights = '/home-ex/tclhk/xie/ADC-code/classification/models/test1226/model_ckpt_40.pth'
crop_size = 224
if os.path.exists(confusion_img_path):
    shutil.rmtree(confusion_img_path)
    os.mkdir(confusion_img_path)

with open(val_dat) as f:
    data = f.readlines()

dat = [da.strip().split() for da in data]


def get_models(model_weights):
    model_dict = torch.load(model_weights)
    class_name = model_dict['class_name']

    state_dict = model_dict['net']

    model = ResNet(class_name=class_name)
    model.to('cuda')

    model.load_state_dict(state_dict)
    model.eval()
    return model, class_name


def predict_result(model, file_path, boxes):
    img = cv2.imread(file_path)
    assert img is not None, file_path
    h, w, _ = img.shape
    boxes_str = ' '.join([' '.join(map(str, s)) for s in boxes])
    # print(boxes_str)
    box_index = np.argmax(boxes[:, 0] * (boxes[:, 3] + boxes[:, 4]))
    _, cx, cy, _, _ = boxes[box_index]
    # 使用conf*宽高最大的那个作为预测结果
    x1 = int(cx * w - crop_size / 2)
    y1 = int(cy * h - crop_size / 2)
    x1 = min(max(0, x1), w - crop_size)
    y1 = min(max(0, y1), h - crop_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    img_crop = img[y1:y2, x1:x2, :]
    # cv2.imwrite('{}.jpg'.format(i), img_crop)

    inputs_numpy = np.transpose(img_crop, (2, 0, 1))
    inputs_numpy = np.expand_dims(inputs_numpy.astype(np.float32), 0)

    with torch.no_grad():
        inputs = torch.from_numpy(inputs_numpy / 255)
        inputs = Variable(inputs.to('cuda'), requires_grad=False)

        f, y = model(inputs)
        y = torch.sigmoid(y).detach().cpu().numpy()
        index = np.argmax(y[0])
        conf = y[0, index]
    return img, index, (x1,y1,x2,y2),conf

def draw_and_save_img(img,gt_boxes,pre_box,conf):
    h, w, _ = img.shape
    if gt_boxes is not None and len(gt_boxes)>1:
        for box in gt_boxes:
            for box in boxes:
                _, x, y, ww, hh = box
                x1 = int((x - (ww / 2)) * w)
                y1 = int((y - (hh / 2)) * h)
                x2 = int((x + (ww / 2)) * w)
                y2 = int((y + (hh / 2)) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    if pre_box is not None and len(pre_box)>1:
        x1, y1, x2, y2= pre_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        xx = min(x1, w - 80)
        yy = y1 - 5 if y1 > 30 else y2 + 25
        if y1 < 30 and y2 > h - 25: yy = h // 2
        cv2.putText(img, '{:.2f}'.format(conf), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, \
                    0.85, (0, 0, 255), 1)
    return img


model, class_name = get_models(model_weights)
Mat = np.zeros((len(class_name),len(class_name)),np.int32)
class_dict = {name: idx for idx, name in enumerate(class_name)}
for mess in tqdm(dat):
    file_path = mess[0]
    c = mess[1]
    if c not in class_name:
        continue
    gt_id = class_dict[c]
    boxes = np.asarray(mess[2:], dtype=np.float32).reshape(-1, 5)
    img, pre_id, box,conf = predict_result(model, file_path, boxes)
    pre_class_name = class_name[pre_id]
    Mat[gt_id,pre_id]+=1
    if confusion_img_path:
        class_path=os.path.join(confusion_img_path,c)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        if gt_id!=pre_id:
            each_class = os.path.join(class_path,class_name[pre_id])
            if not os.path.exists(each_class):
                os.mkdir(each_class)
            img = draw_and_save_img(img,boxes,box,conf)
            cv2.imwrite(os.path.join(each_class,os.path.basename(file_path)), img)

print(Mat)
print('class  {:<7}{:<7}{:<7}{:<7}{:<7}{:<7}'.format(*class_name))
for name, dat in zip(class_name, Mat):
    prstr = ''
    prstr += '{:<7}'.format(name)
    prstr += '{:<7}{:<7}{:<7}{:<7}{:<7}{:<7}'.format(*dat)
    # prstr += '{:.4f}   {:.4f}'.format(dat[0] / dat[1], dat[0] / dat[2])
    print(prstr)
colsum = np.sum(Mat, axis=0).tolist()
# 每一列的和
rowsum = np.sum(Mat, axis=1).tolist()
# 每一行的和
total = np.sum(colsum, axis=0)
diag = np.trace(Mat)
recall = np.diagonal(Mat) / rowsum
precision = np.diagonal(Mat) / colsum
print('class   recall precition')
for name,rec,pre in zip(class_name,recall,precision):
    print('{:<8}{:<7.4f}{:<7.4f}'.format(name,rec,pre))

print('total acc {}'.format(diag/total))
