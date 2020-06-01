OUTPUT_PATH=/data1/chenww/my_research/yolo_v3_x/models/pcb/0508_pcb_ori_bs4_1024_768/pred_result_ep99_conf0.4_nms0.5/
rm -rf ${OUTPUT_PATH}
mkdir ${OUTPUT_PATH}

python predict.py \
--pretrained_weights=/data1/chenww/my_research/yolo_v3_x/models/pcb/0508_pcb_ori_bs4_1024_768/yolov3_ckpt_99.pth  \
--test_data_path=/data1/chenww/my_research/yolo_v3_x/config/pcb/val.dat \
--conf_thres=0.4 \
--iou_thres=0.5 \
--output_txt=${OUTPUT_PATH} \
--image_example=${OUTPUT_PATH}'vis/'  \
2>&1 | tee  ${OUTPUT_PATH}"log.txt" &
