SAVE_PATH=/data1/chenww/my_research/yolo_v3_x/models/pcb/0509_pcb_ori_bs2_1024_768_noobjscale10/
rm -rf ${SAVE_PATH}
mkdir ${SAVE_PATH}

nohup python train.py \
--data_config=./config/pcb/adc.data \
--epochs=200 \
--batch_size=2 \
--n_cpu=0 \
--evaluation_interval=1  \
--debug=False \
--save_path=${SAVE_PATH}  \
--lr=0.01 \
2>&1 | tee  ${SAVE_PATH}"log.txt" &



