SAVE_PATH=/home/chenww/project/Two-Stage-Defect-Detection/detector/models/9cls/9cls_3_train_test_bs8_imgsz7681024_ep150_scratch/
rm -rf ${SAVE_PATH}
mkdir ${SAVE_PATH}

# nohup python train.py \
python train_backup.py \
--data_config=./config/9cls/0107_3_train_test/adc.data \
--epochs=150 \
--batch_size=8 \
--n_cpu=0 \
--evaluation_interval=2  \
--debug=False \
--save_path=${SAVE_PATH}  \
--lr=0.01 \
2>&1 | tee  ${SAVE_PATH}"log.txt" &



