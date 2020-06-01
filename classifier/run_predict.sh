RESULT_DIR=/home-ex/tclhk/chenww/t2/models/classification_x/0103_v3/result_ep30/

rm -rf ${RESULT_DIR}
mkdir ${RESULT_DIR}

#python val_predict.py \
python predict.py \
--pretrained_weights=/home-ex/tclhk/chenww/t2/models/classification_x/0103_v3/model_ckpt_30.pth  \
--result_dir=${RESULT_DIR}  \
--test_data=./data/v3/val.dat \
--debug=0 \
2>&1 | tee  ${RESULT_DIR}"confu_mat.txt"  &
