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

# 用于查看具体每个类的recall的precition，并生成结果和保存混淆图片
# FAS类的处理，无检测框就认为是FAS
debug = True
crop_size = 224
val_data_root = '/home-ex/tclhk/xie/ADC-code/classification/data/val.dat'
confusion_img_path = '/home-ex/tclhk/xie/ADC-code/classification/data/vis/'
model_weights = '/home-ex/tclhk/xie/ADC-code/classification/models/test/model_ckpt_10.pth'
result_save_path = '/home-ex/tclhk/xie/ADC-code/classification/data/result.dat'
gl = open(result_save_path,'w')
with open(val_data_root, 'r') as f:
    data = f.readlines()
lines = [s.strip().split() for s in data]
print(len(lines))


def get_models(model_weights):
    model_dict = torch.load(model_weights)
    class_name = model_dict['class_name']

    state_dict = model_dict['net']

    model = ResNet(class_name=class_name)
    model.to('cuda')

    model.load_state_dict(state_dict)
    model.eval()
    return model, class_name

def draw_gt_pre_box(img,pre_box,gt_box):
    h, w, _ = img.shape
    if len(pre_box)>1:
        pre_name,conf,(x1, y1, x2, y2) = pre_result
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        xx = min(x1, w - 80)
        yy = y1 - 5 if y1 > 30 else y2 + 25
        if y1 < 30 and y2 > h - 25: yy = h // 2
        cv2.putText(img, '{:.2f}'.format(conf), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, \
                    0.85, (0, 0, 255), 1)
    if len(gt_box[2:])>0:
        boxes = np.asarray(gt_box[2:], dtype=np.float32).reshape(-1, 5)
        for box in boxes:
            _,x, y, ww, hh = box
            x1 = int((x-(ww/2)) * w)
            y1 = int((y-(hh/2)) * h)
            x2 = int((x+(ww/2)) * w)
            y2 = int((y+(hh/2)) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return img

model, class_name = get_models(model_weights)
pre_class_dict = {}
test_files = []
for mess in tqdm(lines):
    file_path = mess[0]
    c = mess[1]
    if c not in class_name:
        continue
    boxes = np.asarray(mess[2:], dtype=np.float32).reshape(-1, 5)
    if len(mess) < 3:
        pre_class_dict[file_path] = ('TSFAS')
    test_files.append((file_path, c, boxes))
print('number of predict TSFAS {}'.format(len(pre_class_dict)))
print(class_name)
class_dict = {name: i for i, name in enumerate(class_name)}
print(class_dict)
for image_file, label, boxes in tqdm(test_files):
    if image_file in pre_class_dict:
        continue
    img = cv2.imread(image_file)
    assert img is not None, image_file
    h, w, _ = img.shape
    boxes_str = ' '.join([' '.join(map(str, s)) for s in boxes])
    # print(boxes_str)
    index = np.argmax(boxes[:, 0] * (boxes[:, 3] + boxes[:, 4]))
    _, cx, cy, _, _ = boxes[index]
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
        prename = class_name[index]
        conf = y[0, index]
        pre_class_dict[image_file] = (prename, conf, (x1, y1, x2, y2))


label_files_path = '/home-ex/tclhk/xie/ADC-code/classification/data/val.dat'
for name,value in pre_class_dict.items():
    if len(value)>1:
        w_str = '{} {} {} {} {} {} {}\n'.format(name,value[0],value[1],value[2][0],value[2][1],value[2][2],value[2][3])
    else:
        w_str = '{} {} {}\n'.format(name,value[0],1.0)
    gl.writelines(w_str)
gl.close()
print('total samples {}'.format(len(test_files)))
Mat = np.zeros((len(class_name), len(class_name)),dtype=np.int32)
with open(label_files_path) as fl:
    data = fl.readlines()
dat = [da.strip().split() for da in data]
for mess in tqdm(dat):
    file_path = mess[0]
    label = mess[1]
    if label not in class_name:
        continue
    pre_result = pre_class_dict[file_path]
    Mat[class_dict[label],class_dict[pre_result[0]]]+=1
    True_class_path = os.path.join(confusion_img_path,label)
    if not os.path.exists(True_class_path):
        os.mkdir(True_class_path)
    if label!=pre_result[0]:
        pre_class_path = os.path.join(True_class_path,pre_result[0])
        if not os.path.exists(pre_class_path):
            os.mkdir(pre_class_path)
        img = cv2.imread(file_path)
        draw_gt_pre_box(img,pre_result,mess)
        save_path = os.path.join(pre_class_path,os.path.basename(file_path))
        cv2.imwrite(save_path, img)
print(Mat)


