EVAL_SAVE_PATH=/data1/chenww/t2/models/combine/85/0508_finetuned13d10rotate90_det_ep21_conf0.4_nms0.5_cls_ep99/
EVAL_VIS_PATH=/data1/chenww/t2/models/combine/85/0508_finetuned13d10rotate90_det_ep21_conf0.4_nms0.5_cls_ep99/vis/
rm -rf ${EVAL_SAVE_PATH} ${EVAL_VIS_PATH}
mkdir ${EVAL_SAVE_PATH}  ${EVAL_VIS_PATH}

python pipeline.py \
--det_model_weight=/data1/chenww/t2/models/yolo_v3_x/85/0427_0418_unchecked_data_fintunefrom_d13+d10_rotate90/yolov3_ckpt_21.pth  \
--det_conf_th=0.4  \
--det_nms_th=0.5  \
--cls_model_weight=/data1/chenww/t2/models/classification_x/85/0506_data0418_0422_check_finetune_d13+d10rotate90/model_ckpt_99.pth  \
--eval_img_path=/data1/chenww/dataset/85/0418_0422_check/val.dat \
--eval_gt_path=/data1/chenww/dataset/85/0418_0422_check/val.dat \
--result_save_path=${EVAL_SAVE_PATH}   \
--eval_vis_path=${EVAL_VIS_PATH}  \
2>&1 | tee ${EVAL_SAVE_PATH}"log.txt"

EVAL_SAVE_PATH=/data1/chenww/t2/models/combine/85/0508_finetuned13d10rotate90_det_ep36_conf0.4_nms0.5_cls_ep82/
EVAL_VIS_PATH=/data1/chenww/t2/models/combine/85/0508_finetuned13d10rotate90_det_ep21_conf0.4_nms0.5_cls_ep82/vis/
rm -rf ${EVAL_SAVE_PATH} ${EVAL_VIS_PATH}
mkdir ${EVAL_SAVE_PATH}  ${EVAL_VIS_PATH}

python pipeline.py \
--det_model_weight=/data1/chenww/t2/models/yolo_v3_x/85/0427_0418_unchecked_data_fintunefrom_d13+d10_ori/yolov3_ckpt_36.pth  \
--det_conf_th=0.4  \
--det_nms_th=0.5  \
--cls_model_weight=/data1/chenww/t2/models/classification_x/85/0506_data0418_0422_check_finetune_d13+d10ori90/model_ckpt_82.pth  \
--eval_img_path=/data1/chenww/dataset/85/0418_0422_check/val.dat \
--eval_gt_path=/data1/chenww/dataset/85/0418_0422_check/val.dat \
--result_save_path=${EVAL_SAVE_PATH}   \
--eval_vis_path=${EVAL_VIS_PATH}  \
2>&1 | tee ${EVAL_SAVE_PATH}"log.txt"