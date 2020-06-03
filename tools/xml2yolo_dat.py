import os
import os.path as osp
import xml.dom.minidom as minidom
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2
import pdb

from random import shuffle


'''
function:  从每张图片对应的xml整理成一个.dat文件给检测模型
文件的格式为：[图片绝对路径 cls 1 cx cy fw fh  1 cx cy fw fh  ...]
'''

img_root = '/data1/chenww/dataset/t2/small_8cls/test/'
output_dat_path = '/data1/chenww/my_research/Two-Stage-Defect-Detection/detector/config/small_8cls/'
train_or_val = 'test' # 生成'train.dat或生成val.dat'

def read_xml(xml_filename):
    dom = minidom.parse(xml_filename)
    root = dom.documentElement
    assert (len(root.getElementsByTagName('filename')) == 1)
    assert (len(root.getElementsByTagName('size')) == 1)

    for filename in root.getElementsByTagName('filename'):
        filename = filename.firstChild.data

    # for c in root.getElementsByTagName('folder'):
    #     cls = c.firstChild.data

    # for size in root.getElementsByTagName('size'):
    #     width = size.getElementsByTagName('width')[0].firstChild.data
    #     height = size.getElementsByTagName('height')[0].firstChild.data
    #     depth = size.getElementsByTagName('depth')[0].firstChild.data
    #     # print(width, height, depth)

    label_name_list = []
    for i, label_name in enumerate(root.getElementsByTagName('name')):
        ln = label_name.firstChild.data
        label_name_list.append(ln)

    bboxes = []
    for bndbox in root.getElementsByTagName('bndbox'):
        xmin = bndbox.getElementsByTagName('xmin')[0].firstChild.data
        ymin = bndbox.getElementsByTagName('ymin')[0].firstChild.data
        xmax = bndbox.getElementsByTagName('xmax')[0].firstChild.data
        ymax = bndbox.getElementsByTagName('ymax')[0].firstChild.data
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        bboxes.append((xmin, ymin, xmax, ymax))
    # return filename, bboxes, cls
    return  bboxes, label_name_list


if not os.path.exists(output_dat_path):
    os.makedirs(output_dat_path)

class_list = []
info_list = []
tsfas_list = []

for i, class_name in enumerate(os.listdir(img_root)):
    for img_file in os.listdir(osp.join(img_root, class_name)):
        if img_file.endswith('.xml'):
            continue

        xml = img_file[:-3] + 'xml'
        xml_filename_path = os.path.join(img_root,class_name, xml)
        img_file_path = os.path.join(img_root, class_name, img_file)

        #-----------------------filter-------------------------
        if class_name == 'TSFAS':
            tsfas_list.append((img_file_path))
            continue  # 遇到TSFAS返回，不读取标注文件
        #-----------------------filter-------------------------

        try:  # 防止 xml中的 bboxes信息为空或者没有xml这个文件，会发生错误
            # filename, bboxes, cls = read_xml(xml_filename_path)
            bboxes, label_name_list = read_xml(xml_filename_path)
        except:
            print('xml is none or bbox in xml is none:', xml)
            continue

        fi = os.path.join(img_root, class_name, img_file)
        img = cv2.imread(img_file_path)
        height, width, _ = img.shape
        assert img is not None, img_file_path

        output_txt = ''
        b = ''
        for xmin, ymin, xmax, ymax in bboxes:
            bw = xmax - xmin
            bh = ymax - ymin
            bcx = (xmin + xmax) / 2.0 - 1
            bcy = (ymin + ymax) / 2.0 - 1

            dw = 1. / width
            dh = 1. / height

            x = bcx * dw
            y = bcy * dh
            w = bw * dw
            h = bh * dh
            # b += '0 {} {} {} {} {} '.format(cx, cy, fw, fh, class_name)
            b += '0 {} {} {} {} '.format(x, y, w, h)
        info_list.append((fi, class_name, b))


# shuffle(info_list)
num_file = len(info_list) + len(tsfas_list)
print("len of tsfas:", len(tsfas_list))
print("len of others:", len(info_list))
print("num_file:", num_file)

with open(output_dat_path + train_or_val + '.txt', 'w') as f:
    for img_f, cls, box in info_list:
            f.write('{} {} {}\n'.format(img_f, cls, box))
    # for img_f in tsfas_list:
    #     f.write('{} TSFAS\n'.format(img_f))

print('done')