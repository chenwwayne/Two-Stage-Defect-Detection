#SAVE_PATH=/data1/chenww/my_research/Two-Stage-Defect-Detection/detector/models/small_8cls/1cls_896_bs_ep300_scratch_kmeansAnchor/
SAVE_PATH=/data1/chenww/my_research/Two-Stage-Defect-Detection/detector/models/small_8cls/temp/
rm -rf ${SAVE_PATH}
mkdir ${SAVE_PATH}

nohup python train.py \
--data_config=./config/small_8cls/adc.data \
--epochs=300 \
--batch_size=8 \
--n_cpu=0 \
--evaluation_interval=1  \
--debug=True \
--save_path=${SAVE_PATH}  \
--lr=0.01 \
2>&1 | tee  ${SAVE_PATH}"log.txt" &



