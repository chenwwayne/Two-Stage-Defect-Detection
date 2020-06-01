import os
import cv2
import numpy as np
from random import shuffle

file_list = ['../../dataset/1350/1350.dat', 
             '../../dataset/1350/TEST_BATCH_2_bbox.dat', 
             '../../dataset/1350/1350_SAD.dat',
            ]

image_list = []
class_list = []
for dataset_path in file_list:
    dataset_path = os.path.abspath(dataset_path)
    print(dataset_path)
    
    with open(dataset_path) as f:
        lines = f.readlines()
           
    for line in lines:
        line = line.strip().split()
        filename = line[0]
        classname = line[1]
        class_list.append(classname)
        if classname in ['TSFAS']:
            continue
        if len(line) < 3:
            continue

        bbox = np.asarray(line[2:], dtype=np.str)
        bbox = bbox.reshape(-1, 5)
        conf = bbox[:, 0].astype(np.float32)
        w = bbox[:,3].astype(np.float32)
        h = bbox[:,4].astype(np.float32)
        index = np.argmax(conf*(w + h))
        # ?
        bbox = bbox[index]
        bbox = ' '.join(bbox.tolist())

        image_list.append((filename, classname, bbox))
shuffle(image_list)
num_file = len(image_list)
print(num_file)
print(set(class_list))

num_file = len(image_list)
num_test_file = num_file//5
train_dataset = image_list[num_test_file:]
test_dataset = image_list[:num_test_file]
out_list = [
            ('data.dat', image_list),
            ('train.dat', train_dataset),
            ('test.dat', test_dataset),                
           ]
for path, data in out_list:
    with open(path, 'w') as f:
        for d in data:
            f.write('{} {} {}\n'.format(*d))

print('done')
