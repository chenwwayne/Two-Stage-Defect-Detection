import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from random import shuffle
from collections import OrderedDict
from imgaug import augmenters as iaa
from remove_blacklist import remove_blacklist
from tqdm import tqdm

def crop(img, cx, cy, h, w, crop_size, random=True):
    if random:
        sz = np.random.randint(-30, 30) + crop_size
        x1 = np.random.randint(-80, 80) + cx - sz // 2
        y1 = np.random.randint(-80, 80) + cy - sz // 2

    else:
        x1 = cx - crop_size // 2
        y1 = cy - crop_size // 2
        sz = crop_size

    x1 = min(max(0, x1), w - sz)
    y1 = min(max(0, y1), h - sz)

    img_crop = img[y1:y1 + sz, x1:x1 + sz, :]
    return cv2.resize(img_crop, (crop_size, crop_size))


def augment_sequential():
    return iaa.Sequential([
        iaa.SomeOf((1, 3), [
            iaa.Fliplr(0.5),

            iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=(0.0, 1.0)),

            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),
                iaa.AverageBlur(k=(2, 3)),
            ]),

            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.0)),

            iaa.SimplexNoiseAlpha(iaa.OneOf([
                iaa.EdgeDetect(alpha=(0.0, 0.5)),
                iaa.DirectedEdgeDetect(alpha=(0.0, 0.5), direction=(0.5, 1.0)),
            ])),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            iaa.OneOf([
                iaa.Dropout((0.01, 0.3), per_channel=0.5),
                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.1), per_channel=0.2),
            ]),

            iaa.Add((-50, 50), per_channel=0.5),

            iaa.AddToHueAndSaturation((-50, 50)),

            iaa.LinearContrast((0.8, 1.2), per_channel=0.5),

            iaa.Grayscale(alpha=(0.0, 1.0)),

            iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25),

            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
        ])
    ])


class ListDataset(Dataset):
    def __init__(self, list_file, augment=True, crop_size=224, random_crop=True, \
                 min_data_count=100, class_list=None, debug=False):
        self.debug = debug
        self.augment = augment
        self.crop_size = crop_size
        self.random_crop = random_crop

        if augment:
            self.augment_sequential = augment_sequential()

        with open(list_file, 'r') as f:
            lines = f.readlines()
        # lines = remove_blacklist(lines, None)
        shuffle(lines)
        if debug: lines = lines[:1000]

        img_files = [s.strip().split() for s in lines]

        img_files = np.asarray(img_files)
        classes = img_files[:, 1].copy()
        classlist = classes.tolist()
        class_name = list(set(classlist))

        self.class_list = OrderedDict()
        class_counts = OrderedDict()
        if class_list is not None:
            for name, (i, _) in class_list.items():
                count = classlist.count(name)

                classes[classes == name] = ''
                self.class_list[name] = (i, count)

                class_counts[name] = (i, count)
            others_names = [name for name in class_name if name not in class_counts]
            others_counts = [classlist.count(name) for name in others_names]
            for idx in np.argsort(-np.asarray(others_counts)):
                class_counts[others_names[idx]] = ('', others_counts[idx])

        else:
            
            counts = [classlist.count(name) for name in class_name]

            for i, idx in enumerate(np.argsort(-np.asarray(counts))):

                name = class_name[idx]
                count = counts[idx]
                class_counts[name] = (i, count)

                if count >= min_data_count:
                    classes[classes == name] = ''

                    self.class_list[name] = (i, count)

        indxe = classes == ''

        self.img_files = img_files[indxe, :].copy()
        self.num_files = len(self.img_files)
        self.num_class = len(self.class_list)

        print('Number sample:', self.num_files)
        print('Number class :', self.num_class)
        flag = True
        for key, (i, count) in class_counts.items():
            if flag and key not in self.class_list:
                flag = False
                print('-------------------------')
            print('{:<10}{:<10}{:<10}'.format(key, i, count))
        print()

    def get_data(self, i):
        img_file = self.img_files[i]
        img = cv2.imread(img_file[0])

        # fusion
        if self.augment and np.random.random() < 0.3:
            fusion_rate = np.random.random() / 2
            fusion_id = np.random.randint(1, self.num_files)
            fusion_id = fusion_id if fusion_id > i else fusion_id - 1
            fusion_img = cv2.imread(self.img_files[fusion_id][0]).astype(np.float32)
            img = fusion_rate * fusion_img + (1 - fusion_rate) * (img.astype(np.float32))

            img = img.astype(np.uint8)

        h, w, _ = img.shape

        cx, cy = img_file[3:5].astype(np.float32)

        class_name = img_file[1]
        img = crop(img, int(cx * w), int(cy * h), h, w, self.crop_size, random=self.random_crop)
        if self.augment and np.random.random() < 0.5:

            img = np.expand_dims(img, 0)
            img = self.augment_sequential(images=img)[0]

        label, _ = self.class_list[class_name]
        return img, label

    def __getitem__(self, index):
        i = index % len(self)
        img, label = self.get_data(i)
        # if img.shape[0] > img.shape[2]:
        #     img = img.transpose((2, 0, 1))
        img = transforms.ToTensor()(img)
        return img, label

    def collate_fn(self, batch):
        imgs, label = list(zip(*batch))

        imgs = torch.stack(imgs)
        label = torch.LongTensor(label).view(-1, 1)

        return imgs, label

    def __len__(self):
        return self.num_files


class DataLoader(object):
    def __init__(self, list_path, **kwargs):
        debug = kwargs.pop('debug', False)
        augment = kwargs.pop('augment', False)
        batch_size = kwargs.pop('batch_size', 1)
        crop_size = kwargs.pop('crop_size', 224)
        random_crop = kwargs.pop('random_crop', False)
        min_data_count = kwargs.pop('min_data_count', 100)
        class_list = kwargs.pop('class_list', None)

        self.dataset = ListDataset(list_path, augment=augment, crop_size=crop_size, random_crop=random_crop,
                                   min_data_count=min_data_count, class_list=class_list, debug=debug)
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            collate_fn=self.dataset.collate_fn,
            batch_size=batch_size,
            **kwargs
        )

        self.idx = 0
        self.iter = iter(self.loader)
        self.num_class = self.dataset.num_class
        self.class_name = [key for key, (i, _) in self.dataset.class_list.items()]

    def next(self):
        if self.idx == len(self):
            self.iter = iter(self.loader)
            self.idx = 0
        self.idx += 1
        imgs, label = next(self.iter)

        return imgs, label

    def __len__(self):
        return len(self.loader)

    def save_example(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dataset = self.dataset
        for i in tqdm(range(dataset.num_files)):
            img_file = dataset.img_files[i]
            img_file_name = os.path.split(img_file[0])[-1]
            class_name = img_file[1]
            img, _ = dataset.get_data(i)
            class_path = os.path.join(save_path, class_name)
            if not os.path.exists(class_path):
                os.mkdir(class_path)
            img_path = os.path.join(class_path, img_file_name)
            cv2.imwrite(img_path, img)

    def save_label_name(self, save_path):
        class_list = OrderedDict()
        assert self.dataset.class_list is not None
        for name, (label, _) in self.dataset.class_list.items():
            class_list[label] = name
        with open(save_path, 'w') as f:
            f.write(str(class_list))


if __name__ == '__main__':
    train_files = '../dataset/1350_train.dat'
    test_files = '../dataset/1350_test.dat'
    train_loader = DataLoader(train_files, debug=False, shuffle=True, \
                              batch_size=32, min_data_count=90, augment=True)
    class_list = train_loader.dataset.class_list
    test_loader = DataLoader(test_files, debug=False, shuffle=True, \
                             batch_size=32, class_list=class_list, augment=False)
    train_example = 'models/test/train_example'
    test_example = 'models/test/test_example'
    # p = train_loader.dataset.p
    # print(set(p))
    # exit()

    # if not os.path.exists(train_example):
    #     os.makedirs(train_example)
    # if not os.path.exists(test_example):
    #     os.makedirs(test_example)
    #
    # train_loader.save_example(train_example)
    # test_loader.save_example(test_example)
    # exit()

    for i in range(len(train_loader)):
        imgs, label = train_loader.next()
    dis = train_loader.distribution()
    for ii in dis:
        print(ii)
        # print(imgs)
        # print(label.size())
        #
        # imgs, label = test_loader.next()
        # print(imgs)
        # print(label.size()))
