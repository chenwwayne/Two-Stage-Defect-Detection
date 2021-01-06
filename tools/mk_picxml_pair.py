import os
import os.path as osp
import shutil 
import glob

dataset_root = "/home/chenww/dataset/ALL_2253_D13_IMGXML_PAIR/"


cnt =0
for clsname in os.listdir(dataset_root):
    file_pre_dict = []
    if clsname == 'TSFAS': continue
    for f in os.listdir(osp.join(dataset_root, clsname)):
        file_pre_dict.append(f[:-4])       

    file_pre_set = set(file_pre_dict)

    for pre in file_pre_set:
        temp = glob.glob(dataset_root + clsname + '/' + pre + '*')
        if len(temp) == 1:
            cnt += 1
            mv_path = '/home/chenww/dataset/ALL_2253_D13_SIGNAL_FILE/' + clsname
            if not os.path.exists(mv_path):
                os.makedirs(mv_path)
            shutil.move(osp.join(dataset_root, clsname, temp[0]), mv_path)
print(cnt)

