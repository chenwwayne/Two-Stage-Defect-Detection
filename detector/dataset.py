import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from remove_blacklist import remove_blacklist

def augment_sequential():
    return iaa.Sequential([
        iaa.SomeOf((0, 3), [  # 每次使用0~3个Augmenter来处理图片

            iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=(0.0, 1.0)),  # 边缘检测，只检测某些方向的

            iaa.OneOf([  # 每次以下Augmenters中选择一个来变换
                iaa.GaussianBlur((0, 1.0)),
                iaa.AverageBlur(k=(2, 3)),
            ]),

            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.0)),  # 锐化

            iaa.SimplexNoiseAlpha(iaa.OneOf([
                iaa.EdgeDetect(alpha=(0.0, 0.5)),
                iaa.DirectedEdgeDetect(alpha=(0.0, 0.5), direction=(0.5, 1.0)),
            ])),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            iaa.OneOf([
                iaa.Dropout((0.01, 0.3), per_channel=0.5),  # 随机丢弃像素
                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.1), per_channel=0.2),  # 随机丢弃某位置某通道像素
            ]),

            iaa.Add((-50, 50), per_channel=0.5),  # 像素值成比例增加/减小（特指亮度）

            iaa.AddToHueAndSaturation((-50, 50)),  # 增加色相、饱和度

            iaa.LinearContrast((0.8, 1.2), per_channel=0.5),

            iaa.Grayscale(alpha=(0.0, 1.0)),
        ])
    ])


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=[416, 416], augment=True, multiscale=True):
        with open(list_path, "r") as file:
            lines = file.readlines()
        # lines = remove_blacklist(lines, blacklist_txt='config/blacklist_detection.dat')
        random.shuffle(lines)
        self.img_files = [s.strip().split() for s in lines]

        self.max_objects = 100
        self.multiscale = multiscale
        self.img_size = img_size
        self.augment = augment
        if augment:
            self.augment_sequential = augment_sequential()


    def __getitem__(self, index):
        img_str = self.img_files[index % len(self.img_files)]
        img = cv2.imread(img_str[0])
        class_name = img_str[1]

        if len(img_str) > 2:
            boxes = np.asarray(img_str[2:], dtype=np.float32)
            boxes = boxes.reshape(-1, 5)

            targets = np.zeros((len(boxes), 5), dtype=np.float32)
            targets[:, 1:] = boxes[:,1:]
        else:
            targets = None
        if self.augment:
            if np.random.choice([True, False]):
                img = img[::-1, :, :].copy()
                if targets is not None:
                    targets[:, 1] = 1 - targets[:, 1]
            if np.random.choice([True, False]):
                img = img[:, ::-1, :].copy()
                if targets is not None:
                    targets[:, 0] = 1 - targets[:, 0]
            # 上下左右的随机翻转

            img = self.augment_sequential(image = img)

        return img, targets, (class_name, img_str[0])

    def collate_fn(self, batch):
        imgs, ts, c = list(zip(*batch))
        # print([c_[0] for c_ in c])
        targets = []
        for i, boxes in enumerate(ts):
            if boxes is not None:
                boxes[:, 0] = i
                targets.append(boxes.copy())
        if len(targets) > 0:
            targets = torch.from_numpy(np.concatenate(targets, 0))
        else:
            targets = torch.empty(0)

        h, w = self.img_size
        if self.multiscale:
            offset = random.choice(range(-128, 128, 32))
            h += offset
            w += offset

        # 可以看到这里是直接的resize，宽高比被改变了
        imgs = np.stack([cv2.resize(img, (w, h)) for img in imgs])
        imgs = imgs.transpose(0, 3, 1, 2).astype(np.float32)/255
        imgs = torch.from_numpy(imgs)
        return imgs, targets, c

    def __len__(self):
        return len(self.img_files)
