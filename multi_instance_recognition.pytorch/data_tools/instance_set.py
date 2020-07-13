from torch.utils.data import Dataset, DataLoader
from data_tools.icdar15_loader import ICDAR15Loader
from data_tools.icdar13_loader import ICDAR13Loader
from data_tools.SynthText_loader import SynthTextLoader
from data_tools.data_utils import check_and_validate_polys, crop_box
from data_tools.image_list import to_image_list
from .data_path import DATAPATH
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from mmcv.parallel.data_container import  DataContainer
from torchvision.transforms import functional as F
import random


class ImgInstanceLoader(object):

    def __init__(self, cfg):
        self.img_dir = cfg['data_dir']
        self.gt_path = cfg['gt_dir']
        self.data_type = cfg['data_type']
        self.num_instances = cfg['num_instances']
        # self.input_width, self.input_height = cfg['input_width'], cfg['input_height']
        self.max_len = cfg['max_len']
        self.batch_size = cfg['batch_size']
        self.num_works = cfg['num_works']
        self.shuffle = ['shuffle']
        self.instance_set = InstanceSet(cfg)

        self.loader = DataLoader(self.instance_set, self.batch_size, self.shuffle,
                                 collate_fn=my_collate, num_workers=self.num_works, pin_memory=True)

    def __len__(self):
        return len(self.instance_set)

    def __iter__(self):
        return iter(self.loader)


class InstanceSet(Dataset):

    def __init__(self, cfg, data_type):
        '''
        :param data_dir:
        :param gt_dir:
        :param data_type: "SynthText" or "ICDAR"
        :param max_len:
        '''
        # self.data_dir = data_dir
        # self.gt_dir = gt_dir
        # self.data_type = data_type
        # self.max_len = max_len
        # self.input_width = cfg['input_width']
        # self.input_height = cfg['input_height']
        self.num_instances = cfg['num_instances']
        self.transforms = ResizeNormalize(cfg['size'])
        self.crop_ratio = cfg['crop_ratio']
        self.crop_random = cfg['crop_random']

        assert data_type in DATAPATH.keys()
        if 'synthtext' in data_type:
            self.data = SynthTextLoader(*DATAPATH[data_type], max_len=cfg['max_len'])
        elif 'icdar2015' in data_type:
            self.data = ICDAR15Loader(*DATAPATH[data_type])
        else:
            self.data = ICDAR13Loader(*DATAPATH[data_type])

    def __len__(self):
        return self.data.num_samples

    def __getitem__(self, index):
        try:
            img, image_path, text_polys, text_tags, text_labels, text_label_masks, words = self.data.get_sample(index)

            if len(text_labels) == 0:
                return None
            W,H = img.size
            text_polys, text_tags, text_labels, text_label_masks, words = check_and_validate_polys(text_polys, text_tags,
                                                                                                   text_labels,
                                                                                                   text_label_masks, words,
                                                                                                   (H, W))

            # img = cv2.resize(img, dsize=(self.input_width, self.input_height))
            # if self.keep_ratio:
            #     new_w = self.input_width
            #     resize_ratio = float(new_w) / img.width
            #     new_h = img.height * resize_ratio
            #     new_h = round(new_h)
            # else:
            #     new_w, new_h = self.input_width, self.input_height
            img, text_polys = self.transforms(img, text_polys)

            # resize_ratio_3_x = new_w / float(W)
            # resize_ratio_3_y = new_h / float(H)
            # text_polys[:, :, 0] *= resize_ratio_3_x
            # text_polys[:, :, 1] *= resize_ratio_3_y

            rectangles = torch.tensor(crop_box(text_polys, num=self.num_instances, iratio=self.crop_ratio, israndom=self.crop_random), dtype=torch.float32) # n * num_instances * 5
            # rectangles[:, :, 1] /= float(self.input_height)
            # rectangles[:, :, 2] /= float(self.input_width)
            # rectangles[:, :, 3] /= float(self.input_height)
            # rectangles[:, :, 4] /= float(self.input_width)
            assert (rectangles.shape[0] * rectangles.shape[1]) == (self.num_instances) * len(text_polys) # n * num_instances *5
            # rectangles = rectangles.reshape(rectangles.shape[0] * rectangles.shape[1], -1) #(N * num_instances) * 5

            # boxes_index = np.ones(shape=[rectangles.shape[0], rectangles.shape[1]], dtype=np.int32) * len(batch_images)

            # batch_images.append(img[:, :, ::-1].astype(np.float32))
            # batch_image_fns.append(img_path)
            # rectangles = torch.tensor(rectangles, dtype=torch.float32)
            text_labels = torch.tensor(text_labels, dtype=torch.long) # n * 25
            # text_labels = text_labels.fill_(10)
            text_label_masks = torch.tensor(text_label_masks, dtype=torch.float32, requires_grad=False)
            return img, image_path, rectangles, text_labels, text_label_masks, words
        except Exception as e:
            print(e)
            return None


class ResizeNormalize(object):

    def __init__(self, size, keep_ratio=True, interpolation=Image.BICUBIC):
        self.min_size = size[0]
        self.max_size = size[1]
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        if isinstance(self.min_size, tuple):
            if len(self.min_size) == 1:
                size = self.min_size[0]
            else:
                random_size_index = random.randint(0, len(self.min_size) - 1)
                size = self.min_size[random_size_index]
        else:
            size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

        # def __call__(self, image, target):
        #     size = self.get_size(image.size)
        #     image = F.resize(image, size)
        #     if target is not None:
        #         target = target.resize(image.size)
        #     return image, target

    def __call__(self, img, text_polys):

        W, H = img.size
        size = self.get_size(img.size)
        img = F.resize(img, size)

        # img = img.resize(size, self.interpolation)
        resize_ratio_3_x = size[1] / float(W)
        resize_ratio_3_y = size[0] / float(H)
        text_polys[:, :, 0] *= resize_ratio_3_x
        text_polys[:, :, 1] *= resize_ratio_3_y
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img, text_polys


def my_collate(batche):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = filter(lambda x:x is not None, batche)
    batch_imgs = []
    batch_imgs_path = []
    batch_rectangles = []
    batch_text_labels = []
    batch_text_labels_mask = []
    batch_words = []
    for i, ii in enumerate(batch):
        batch_imgs.append(ii[0])
        batch_imgs_path.append(ii[1])
        # batch_index = torch.ones(ii[2].shape[0]) * i
        rectangles = ii[2] # n * num_instances * 5
        rectangles[:,:, 0] *= i # batch index
        batch_rectangles.append(rectangles)
        batch_text_labels.append(ii[3])
        batch_text_labels_mask.append(ii[4])
        batch_words.append(ii[5])
    try:
        batch_imgs = to_image_list(batch_imgs).tensors
        batch_rectangles = torch.cat(batch_rectangles)
        batch_text_labels = torch.cat(batch_text_labels)
        batch_text_labels_mask = torch.cat(batch_text_labels_mask)
        # batch_imgs = torch.stack(batch_imgs)
        # batch_rectangles = DataContainer(batch_rectangles) # N * num_instances * 5
        # batch_text_labels = DataContainer(batch_text_labels) # N * 25
        # batch_text_labels_mask = DataContainer(batch_text_labels_mask)
    except Exception as e:
        print(e)
        return None, None, None, None, None, None

    return batch_imgs, batch_imgs_path, batch_rectangles, batch_text_labels, batch_text_labels_mask, batch_words



