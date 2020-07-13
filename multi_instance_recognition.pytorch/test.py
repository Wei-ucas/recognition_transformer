import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import numpy as np
from mmcv import Config
from data_tools.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
import matplotlib.pyplot as plt
import cv2
from mmcv.parallel import MMDataParallel

from model import MultiInstanceRecognition
from data_tools.instance_set import ImgInstanceLoader

torch.backends.cudnn.enabled = False


def test(args, cpks):

    assert isinstance(cpks, str)

    voc, char2id, id2char = get_vocabulary("ALLCASES_SYMBOLS")

    test_data = ImgInstanceLoader(args.val_data_cfg)
    print("test data: {}".format(len(test_data)))

    model = MultiInstanceRecognition(args.model_cfg).cuda()
    # model = MMDataParallel(model).cuda()
    model.load_state_dict(torch.load(cpks))
    model.eval()
    pred_strs = []
    gt_strs = []
    test_data_iter = iter(test_data)
    for i, batch_data in enumerate(test_data):
        torch.cuda.empty_cache()
        batch_imgs, batch_imgs_path, batch_rectangles, \
        batch_text_labels, batch_text_labels_mask, batch_words = \
            batch_data
        if batch_imgs is None:
            continue
        batch_imgs = batch_imgs.cuda()
        batch_rectangles = batch_rectangles.cuda()
        batch_text_labels = batch_text_labels.cuda()
        with torch.no_grad():
            loss, decoder_logits = model(batch_imgs, batch_text_labels, batch_rectangles, batch_text_labels_mask)

        pred_labels = decoder_logits.argmax(dim=2).cpu().numpy()
        pred_value_str = idx2label(pred_labels, id2char, char2id)
        gt_str = batch_words


        for i in range(len(gt_str[0])):
            print("predict: {} label: {}".format(pred_value_str[i], gt_str[0][i]))
            pred_strs.append(pred_value_str[i])
            gt_strs.append(gt_str[0][i])

        val_dec_metrics_result = calc_metrics(pred_strs,
                                              gt_strs, metrics_type="accuracy")

        print("test accuracy= {:3f}".format(val_dec_metrics_result))
        #
        #
        #                                                                         val_loss_value))
        print('---------')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a recognizer")
    parser.add_argument('config', help='config file')
    parser.add_argument('cpks', help='Checkpoint')
    # parser.add_argument('--')
    args = parser.parse_args()
    return args

def plot(img, bbox, name):
    '''

    :param img: c * w* h
    :param bbox: N * n * 5
    :return:
    '''
    img = img.cpu().permute(1,2,0)
    img = (img*0.5 + 0.5)*255
    img = img.long().numpy()
    bbox = bbox.long().cpu().numpy()
    for i in range(bbox.shape[0]):
        for j in range(bbox.shape[1]):
            img = cv2.rectangle(img, (bbox[i,j,1],bbox[i,j,2]), (bbox[i,j,3],bbox[i,j,4]), (0,0,255),1)

    cv2.imwrite('vis/' + name, img)


def plot_instance(img, bbox, labels, name):
    img = img.cpu().permute(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255
    img = img.long().numpy()
    bbox = bbox.long().cpu().numpy()
    if not os.path.exists('vis/'+name):
        os.mkdir('vis/'+name)
    for i in range(bbox.shape[0]):
        for j in range(bbox.shape[1]):
            ins = img[bbox[i,j,2]:bbox[i,j,4],bbox[i,j,1]:bbox[i,j,3] , :]
            cv2.imwrite('vis/'+name + '/' + labels[i]+'_{}.jpg'.format(j),ins)




def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    test(cfg, args.cpks)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()