import os
import os.path as osp
import random
import shutil
from tqdm import tqdm

ratio = 0.9
root = '/home/chenww/dataset/2253_d13_9cls_73/3/'
cp_dst = '/home/chenww/dataset/2253_d13_9cls_73/3_train_test/'

if not osp.exists(cp_dst):
    os.makedirs(cp_dst)


file_pre_list = []
for cls in os.listdir(root):
    for file in os.listdir(osp.join(root, cls)):
        if file.endswith('.jpg'):
            file_pre_list.append([cls, osp.join(root, cls), file[:-4]])

random.shuffle(file_pre_list)
length = len(file_pre_list)

for i, file_pre in enumerate(tqdm(file_pre_list)):
    if i <= length*ratio:
        cls_dst = osp.join(cp_dst, 'train', file_pre[0])
        if not osp.exists(cls_dst): os.makedirs(cls_dst)

        shutil.copy(osp.join(file_pre[1], file_pre[2]+'.jpg'), cls_dst)
        if not file_pre[0] == 'TSFAS':
            shutil.copy(osp.join(file_pre[1], file_pre[2]+'.xml'), cls_dst)
    else:
        cls_dst = osp.join(cp_dst, 'test', file_pre[0])
        if not osp.exists(cls_dst): os.makedirs(cls_dst)

        shutil.copy(osp.join(file_pre[1], file_pre[2]+'.jpg'), cls_dst)
        if not file_pre[0] == 'TSFAS':
            shutil.copy(osp.join(file_pre[1], file_pre[2]+'.xml'), cls_dst)
