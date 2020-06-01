import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import cv2
import json
from yolo_v3_x.model import ResNet as det_resnet
from yolo_v3_x.utils import non_max_suppression
from classification_x.model import ResNet as cla_resnet
import numpy as np
import torch
from torch.autograd import Variable
import shutil
from tqdm import tqdm
import argparse
import sys


class Predict(object):
    def __init__(self, config):
        self.config = config
        self.detect_config = config['detect_config']
        self.classify_config = config['classify_config']
        self.eval_vis_path = config['eval_vis_path']
        self.det_model = self.get_det_model(self.detect_config)
        self.classify_model, self.class_name = self.get_classify_model(self.classify_config)
        self.class_name.append('TSFAS')
        self.detect_result_dict = {}
        # 形式为{image_path:[{bndbox:box,det_conf:score}]},box为解归一化的左上右下
        self.final_resutl_dict = {}
        # 形式为{image_path:{label:name,bndbox:box,conf:sore}}TSFAS类bodbox为[],正常为解归一化的左上右下

    def get_det_model(self, config):
        model_dict = torch.load(config['model_weight'])
        anchors = model_dict['anchors'].to('cuda')
        model = det_resnet(anchors, Istrain=False).to('cuda')
        model.load_state_dict(model_dict['net'])
        model.eval()
        return model

    def inference_det_model(self, img):
        process_img = cv2.resize(img, self.detect_config['process_size'])
        inputs = process_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        inputs = np.expand_dims(inputs, 0)
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs, requires_grad=False).to('cuda')
        with torch.no_grad():
            _, outputs = self.det_model(inputs)
            outputs = non_max_suppression(outputs, conf_thres=self.detect_config['conf_thres'],
                                          nms_thres=self.detect_config['nms_thres'])
            outputs_numpy = []
            for output in outputs:
                if output is None:
                    outputs_numpy.append(None)
                else:
                    outputs_numpy.append(output.detach().cpu().numpy())
        assert len(outputs_numpy) == 1
        return outputs_numpy

    def get_classify_model(self, config):
        model_dict = torch.load(config['model_weight'])
        class_name = model_dict['class_name']

        state_dict = model_dict['net']

        model = cla_resnet(class_name=class_name)
        model.to('cuda')

        model.load_state_dict(state_dict)
        model.eval()
        return model, class_name

    def inference_cla_model(self, img, boxes):
        # img = cv2.imread(file_path)
        assert img is not None
        h, w, _ = img.shape
        # boxes_str = ' '.join([' '.join(map(str, s)) for s in boxes])
        # print(boxes_str)
        # 使用conf*(宽+高)最大的那个作为预测结果,其中，boxes[:, 4] 为分数
        # box_index = np.argmax(boxes[:, 4] * (boxes[:, 2] + boxes[:, 3] - boxes[:, 0] - boxes[:, 1]))
        box_index = np.argmax(boxes[:, 4] * ((boxes[:, 2] - boxes[:, 0])*(boxes[:, 3] - boxes[:, 1])))
        x1_, y1_, x2_, y2_, _ = boxes[box_index]
        cx = (x1_ + x2_) / 2
        cy = (y1_ + y2_) / 2

        crop_size = self.classify_config['crop_size']
        x1 = int(cx * w - crop_size / 2)
        y1 = int(cy * h - crop_size / 2)
        x1 = min(max(0, x1), w - crop_size)
        y1 = min(max(0, y1), h - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        img_crop = img[y1:y2, x1:x2, :]
        # cv2.imwrite('{}.jpg'.format(i), img_crop)

        inputs_numpy = np.transpose(img_crop, (2, 0, 1))
        inputs_numpy = np.expand_dims(inputs_numpy.astype(np.float32), 0)

        with torch.no_grad():
            inputs = torch.from_numpy(inputs_numpy / 255)
            inputs = Variable(inputs.to('cuda'), requires_grad=False)
            f, y = self.classify_model(inputs)
            y = torch.sigmoid(y).detach().cpu().numpy()
            index = np.argmax(y[0])
            conf = y[0, index]
        return index, (x1, y1, x2, y2), conf

    def combine_test(self, file_path):
        img = cv2.imread(file_path)
        det_single_result = self.inference_det_model(img)[0]
        self.detect_result_dict[file_path] = []
        if det_single_result is None:
            inference_box = []
            pre_class_name = 'TSFAS'
            score = 0.3
        else:
            for box in det_single_result:
                box = box.tolist()
                self.detect_result_dict[file_path].append({'bndbox': box[0:4], 'conf': float(box[4])})
            class_id, inference_box, score = self.inference_cla_model(img, det_single_result)
            pre_class_name = self.class_name[class_id]
        return inference_box, pre_class_name, score

    def draw_box_in_img(self, image_path, box_dict):
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        if len(box_dict['bndbox']) > 0:
            x1, y1, x2, y2 = box_dict['bndbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            xx = min(x1, w - 80)
            yy = y1 - 5 if y1 > 30 else y2 + 25
            if y1 < 30 and y2 > h - 25: yy = h // 2
            cv2.putText(img, '{:.2f}'.format(box_dict['conf']), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, \
                        0.85, (0, 0, 255), 1)
        return img


    def eval_with_label(self, gt_dict):
        # 使用标注评估，gt_dict为{image_name:class}
        # assert os.path.exists(self.eval_vis_path)
        if os.path.exists(self.eval_vis_path):
            shutil.rmtree(self.eval_vis_path)
        os.mkdir(self.eval_vis_path)
        confusion_mat = np.zeros((len(self.class_name), len(self.class_name)), np.int32)
        class_to_id = {name: id for id, name in enumerate(self.class_name)}

        # 分数阈值
        conf_ther_list = np.linspace(0, 1, 21)

        print("total val num:",len(gt_dict))
        print('=============accuracy and coverage under different classification thresholds==================')
        print('{:<7}{:<7}{:<7}{:<7}'.format('conf', 'acc', 'ratio', 'a&r'))
        for i, conf_ther in enumerate(conf_ther_list):
            Mat = np.zeros((len(self.class_name), len(self.class_name)), np.int32)
            unconfirm_number = 0
            for image_path, mess in tqdm(self.final_resutl_dict.items()):
                image_name = os.path.basename(image_path)
                gt_name = gt_dict[image_name]
                gt_id = class_to_id[gt_name]
                pre_name = mess['label']
                pre_id = class_to_id[pre_name]
                conf = mess['conf']
                if conf >= conf_ther or pre_name == 'TSFAS':
                    Mat[gt_id][pre_id] += 1
                else:
                    unconfirm_number += 1

                # 计算分类分数阈值为0时的混淆矩阵 及 保存保存分类错误的图片
                if i == 0 and conf >= 0.0:
                    confusion_mat[gt_id][pre_id] += 1
                    if gt_id != pre_id:
                        dir_save_path = os.path.join(self.eval_vis_path, gt_name)
                        if not os.path.exists(dir_save_path):
                            os.mkdir(dir_save_path)
                        save_folder = os.path.join(dir_save_path, pre_name)
                        if not os.path.exists(save_folder):
                            os.mkdir(save_folder)
                        save_path = os.path.join(save_folder, os.path.basename(image_path))
                        save_img = self.draw_box_in_img(image_path, mess)
                        cv2.imwrite(save_path, save_img)

            # 统计不同阈值下的分类准确率、覆盖率、准确率*覆盖率
            colsum = np.sum(Mat, axis=0).tolist()
            # rowsum = np.sum(Mat, axis=1).tolist()
            total = np.sum(colsum, axis=0)
            diag = np.trace(Mat)
            total_acc = diag / total

            confirm_ratio = 1 - unconfirm_number / len(self.final_resutl_dict)
            print('{:<7.4f}{:<7.4f}{:<7.4f}{:<7.4f}'.format(conf_ther, total_acc, confirm_ratio, \
                                                            (total_acc * confirm_ratio)))

        # 计算阈值为设定值后的混淆矩阵
        print("=============confusion matrix with a classification threshold of 0==================")
        print(('class  '+ '{:<7}'*len(self.class_name)).format(*self.class_name))
        for name, dat in zip(self.class_name, confusion_mat):
            prstr = ''
            prstr += '{:<7}'.format(name)
            prstr += ('{:<7}'*len(self.class_name)).format(*dat)
            # prstr += '{:.4f}   {:.4f}'.format(dat[0] / dat[1], dat[0] / dat[2])
            print(prstr)
        colsum = np.sum(confusion_mat, axis=0).tolist()
        # 每一列的和
        rowsum = np.sum(confusion_mat, axis=1).tolist()
        # 每一行的和
        total = np.sum(colsum, axis=0)
        diag = np.trace(confusion_mat)
        recall = np.diagonal(confusion_mat) / rowsum
        precision = np.diagonal(confusion_mat) / colsum
        print('class   recall precition')
        for name, rec, pre in zip(self.class_name, recall, precision):
            print('{:<8}{:<7.4f}{:<7.4f}'.format(name, rec, pre))
        print('total acc {}'.format(diag / total))



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model_weight", type=str, default="")
    parser.add_argument("--det_conf_th", type=float, default=0.5)
    parser.add_argument("--det_nms_th", type=float, default=0.1)

    parser.add_argument("--cls_model_weight", type=str, default="")
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--eval_img_path", type=str, default="")
    parser.add_argument("--result_save_path", type=str, default="")
    parser.add_argument("--eval_gt_path", type=str, default=None)
    parser.add_argument("--eval_vis_path", type=str, default="")

    args = parser.parse_args(argv)

    detect_config = {
        'model_weight': args.det_model_weight,
        'conf_thres': args.det_conf_th, 'nms_thres': args.det_nms_th, 'process_size': (1024, 768)}
    classify_config = {
        'model_weight': args.cls_model_weight,
        'crop_size': args.crop_size}

    config_dict = {'classify_config': classify_config,
                   'detect_config': detect_config,
                   'eval_img_path': args.eval_img_path,
                   'result_save_path': args.result_save_path,
                   'eval_gt_path': args.eval_gt_path,
                   'eval_vis_path': args.eval_vis_path}

    predict = Predict(config_dict)
    img_folder = config_dict['eval_img_path']
    result_save_path = config_dict['result_save_path']
    with open(config_dict['eval_img_path']) as gl:
        data = gl.readlines()
    image_file_list = [da.strip().split()[0] for da in data if da.strip().split()[1] in predict.class_name]
    # image_file_list = []
    # for dir_name, folders, files in os.walk(img_folder):
    #     if len(files) == 0:
    #         continue
    #     for file in files:
    #         if not file.endswith('.jpg'):
    #             continue
    #         image_file_list.append(os.path.join(dir_name, file))
    ################################ conmbine test #####################################
    # for image_path in tqdm(image_file_list[::100]):
    for image_path in tqdm(image_file_list):
        try:
            inference_box, pre_class_name, score = predict.combine_test(image_path)
        except:
            print("combine_test went wrong in pic:", image_path)
            continue
        predict.final_resutl_dict[image_path] = {'label': pre_class_name, 'bndbox': inference_box, 'conf': float(score)}
    # print(predict.detect_result_dict)
    # print(predict.final_resutl_dict)
    ############################### result save #########################################
    detect_save_path = os.path.join(result_save_path, 'detect_result.json')
    final_save_path = os.path.join(result_save_path, 'final_result.json')
    with open(detect_save_path, 'w') as fl:
        json.dump(predict.detect_result_dict, fl)
    with open(final_save_path, 'w') as g:
        json.dump(predict.final_resutl_dict, g)
    print('save success')
    ################################# eval ##############################################
    if config_dict['eval_gt_path'] is not None:
        print('start eval')
        with open(config_dict['eval_gt_path']) as f:
            data = f.readlines()
        gt_dict = {os.path.basename(da.strip().split()[0]): da.strip().split()[1] for da in data if
                   da.strip().split()[1] in predict.class_name}
        predict.eval_with_label(gt_dict)



if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1:])