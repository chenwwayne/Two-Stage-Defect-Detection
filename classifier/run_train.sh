SAVE_PATH=/home-ex/tclhk/chenww/t2/models/classification_x/85/0506_data0418_0422_check_finetune_d13+d10rotate90/
rm -rf ${SAVE_PATH}
mkdir ${SAVE_PATH}

nohup python train.py \
--epochs=200 \
--batch_size=32 \
--train_file=/home-ex/tclhk/chenww/t2/classification_x/data/85/0418_0422_check_data/train.dat  \
--test_file=/home-ex/tclhk/chenww/t2/classification_x/data/85/0418_0422_check_data/val.dat \
--n_cpu=1 \
--evaluation_interval=1  \
--debug=False \
--save_path=${SAVE_PATH}  \
--lr=0.01 \
--augment=False \
--pretrained_weights=/home-ex/tclhk/chenww/t2/models/classification_x/d13+d10/0428_all_rotate90/model_ckpt_199.pth  \
2>&1 | tee  ${SAVE_PATH}"log.txt" &
#--pretrained_weights=/home-ex/tclhk/chenww/t2/models/classification_x/d13/model_ckpt_23.pth  \
#--pretrained_weights=None  \
