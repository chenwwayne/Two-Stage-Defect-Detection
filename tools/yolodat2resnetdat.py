import os
import shutil
import numpy as np
import pdb

dat_path='/data1/chenww/t2/yolo_v3_x/config/27/0813/val.dat'
result_path = '/data1/chenww/t2/classification_x/data/27/0813/'

# pick the maximum bounding box in boxes
def max_box(boxes):
    max_wh = 0
    box_idx = -1
    for idx, box in enumerate(boxes):
        if box[3]+box[4] > max_wh:
            max_wh = box[3]+box[4]
            box_idx = idx

    # return example [ 1.  0.72808909  0.80624998  0.12715517  0.13365385]
    if box_idx==-1:
        print("bbox in xml is none")
        return None
    return boxes[box_idx]


if not os.path.exists(result_path): os.makedirs(result_path)

file = dat_path.split('/')[-1]

with open(dat_path) as f:
    data = f.readlines()

data = [da.strip().split() for da in data]
g = open(result_path + file,'w')
for dat in data:
    file_name = dat[0]
    class_name = dat[1]
    if class_name=='TSFAS':
        continue
    boxes = np.asarray(dat[2:], dtype=np.float32).reshape(-1, 5)
    box = max_box(boxes)
    if box is None: # in some case , code excpet TSFAS got none bounding box
        print(file_name)
        continue
    g.write('{} {} {} {} {} {} {}\n'.format(file_name, class_name, box[0], box[1], box[2], box[3], box[4]))

    # for box in boxes:
    #     g.write('{} {} {} {} {} {} {}\n'.format(file_name, class_name,box[0],box[1],box[2],box[3],box[4]))

g.close()