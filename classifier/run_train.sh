SAVE_PATH=/home/chenww/project/Two-Stage-Defect-Detection/classifier/models/9cls/9cls_3_train_test_bs32/
rm -rf ${SAVE_PATH}
mkdir ${SAVE_PATH}

python3 train.py \
--epochs=150 \
--batch_size=32 \
--train_file=/home/chenww/project/Two-Stage-Defect-Detection/classifier/data/9cls/0107_3_train_test/train.dat  \
--test_file=/home/chenww/project/Two-Stage-Defect-Detection/classifier/data/9cls/0107_3_train_test/test.dat \
--n_cpu=1 \
--evaluation_interval=2  \
--debug=False \
--save_path=${SAVE_PATH}  \
--lr=0.01 \
--augment=False \
2>&1 | tee  ${SAVE_PATH}"log.txt" &
#--pretrained_weights=/home-ex/tclhk/chenww/t2/models/classification_x/d13/model_ckpt_23.pth  \
#--pretrained_weights=None  \
