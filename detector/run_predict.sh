OUTPUT_PATH=/home/chenww/project/Two-Stage-Defect-Detection/detector/models/9cls/9cls_3_train_test_bs8_imgsz7681024_ep150_scratch/example/
rm -rf ${OUTPUT_PATH}
mkdir ${OUTPUT_PATH}

python predict.py \
--pretrained_weights=/home/chenww/project/Two-Stage-Defect-Detection/detector/models/9cls/9cls_3_train_test_bs8_imgsz7681024_ep150_scratch/yolov3_ckpt_14.pth  \
--test_data_path=/home/chenww/project/Two-Stage-Defect-Detection/detector/config/9cls/0107_3_train_test/test.dat \
--conf_thres=0.4 \
--iou_thres=0.5 \
--output_txt=${OUTPUT_PATH} \
--image_example=${OUTPUT_PATH}'vis/'  \
2>&1 | tee  ${OUTPUT_PATH}"log.txt" &
