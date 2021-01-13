import os
import shutil
from remove_blacklist import remove_blacklist

data_txt = './train.dat'
output_dir = 'CLASSIFY_1350_TRAIN'
output_txt = 'file_list_train.dat'

data_txt = './test.dat'
output_dir = 'CLASSIFY_1350_TEST'
output_txt = 'file_list_test.dat'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

with open(data_txt) as f:
    lines = f.readlines()
lines = remove_blacklist(lines)

nfile = len(lines)
error_count = 0
with open(output_txt, 'w') as f:
    for line in lines:
        line = line.strip().split()
        filename = line[0]
        true = line[1]
        pred = line[2]
        conf = float(line[3])
        if true != pred or conf < 0.99:
            base_name = os.path.basename(filename)
            opath = os.path.join(output_dir, true)
            if not os.path.exists(opath):
                os.makedirs(opath)
            shutil.copy(filename, os.path.join(opath, base_name))
            error_count += 1
            f.write('{}\n'.format(os.path.join(true, base_name)))
print(error_count, nfile, error_count/nfile)