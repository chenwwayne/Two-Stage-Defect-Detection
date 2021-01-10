#!/usr/bin/python3

import sys
import prettytable as pt
import os 
import os.path as osp

root = osp.realpath(sys.argv[1]) # 接收路径参数

dataset_stat = []
total_pic_file_cnt = 0
total_xml_file_cnt = 0
tb = pt.PrettyTable(["Code", "Pic_num", "Xml_num"])

for cls_id, cls_name in enumerate(os.listdir(root)):
    pic_file_cnt = 0
    xml_file_cnt = 0
    cls_stat = [cls_name]
    for file in os.listdir(osp.join(root, cls_name)):
        if file.endswith(".jpg"):
            pic_file_cnt += 1
            total_pic_file_cnt += 1
        elif file.endswith(".xml"):
            xml_file_cnt += 1
            total_xml_file_cnt += 1
        else:
            assert 0
    cls_stat.extend([pic_file_cnt, xml_file_cnt])
    tb.add_row(cls_stat)
    
# 添加合计
tb.add_row(['TOTAL', total_pic_file_cnt, total_xml_file_cnt])

tb.sortby = "Pic_num"
tb.reversesort = True



print(tb)
