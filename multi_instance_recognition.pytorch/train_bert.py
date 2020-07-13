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

# from model import MultiInstanceRecognition
from model_bert import ModelBert
from data_tools.instance_set import ImgInstanceLoader

torch.backends.cudnn.enabled = False


def train(args):
    train_data = ImgInstanceLoader(args.train_data_cfg)
    print("train data: {}".format(len(train_data)))
    val_data = ImgInstanceLoader(args.val_data_cfg)
    print("val data: {}".format(len(val_data)))

    model = ModelBert(args.model_cfg).cuda()
    # model = torch.nn.DataParallel(model).cuda()

    if args.resume_from is not None:
        print('loading pretrained models from {opt.continue_model}')
        model.load_state_dict(torch.load(args.resume_from))

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    optimizer = optim.Adam(filtered_parameters, lr=args.lr, betas=(0.9, 0.999))
    lrScheduler = lr_scheduler.MultiStepLR(optimizer, [1, 2, 3], gamma=0.5)

    max_iters = args.max_iters
    start_iter = 0
    if args.resume_from is not None:
        start_iter = int(args.resume_from.split('_')[-1].split('.')[0])
        print('continue to train, start_iter: {start_iter}')

    log_file = open(os.path.join(args.save_name, 'train_log.txt'), 'w')

    train_data_iter = iter(train_data)
    val_data_iter = iter(val_data)
    start_time = time.time()
    for i in range(start_iter, max_iters):
        model.train()
        try:
            batch_data = next(train_data_iter)
        except StopIteration:
            train_data_iter = iter(train_data)
            batch_data = next(train_data_iter)
        data_time_s = time.time()
        batch_imgs, batch_imgs_path, batch_rectangles, \
        batch_text_labels, batch_text_labels_mask, batch_words = \
            batch_data
        while batch_imgs is None:
            batch_data = next(train_data_iter)
            batch_imgs, batch_imgs_path, batch_rectangles, \
            batch_text_labels, batch_text_labels_mask, batch_words = \
                batch_data
        batch_imgs = batch_imgs.cuda()
        batch_rectangles = batch_rectangles.cuda()
        batch_text_labels = batch_text_labels.cuda()
        data_time = time.time() - data_time_s
        # print(time.time() -s)
        # s = time.time()
        loss, decoder_logits = model(batch_imgs, batch_text_labels, batch_rectangles, batch_text_labels_mask)
        # print(time.time() - s)
        # print('------')
        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # print(time.time() - s)
        # print('------')

        if i % args.train_verbose == 0:
            this_time = time.time() - start_time
            log_info = "train iter :{}, time: {:.2f}, data_time: {:.2f}, Loss: {:.3f}".format(i, this_time, data_time*10, loss.data)
            log_file.write(log_info + '\n')
            print(log_info)
            torch.cuda.empty_cache()

        if i % args.val_iter == 0:
            print("--------Val iteration---------")
            model.eval()

            try:
                val_batch = next(val_data_iter)
            except StopIteration:
                val_data_iter = iter(val_data)
                val_batch = next(val_data_iter)

            batch_imgs, batch_imgs_path, batch_rectangles, \
            batch_text_labels, batch_text_labels_mask, batch_words = \
                val_batch
            while batch_imgs is None:
                val_batch = next(val_data_iter)
                batch_imgs, batch_imgs_path, batch_rectangles, \
                batch_text_labels, batch_text_labels_mask, batch_words = \
                    val_batch
            batch_imgs = batch_imgs.cuda()
            batch_rectangles = batch_rectangles.cuda()
            batch_text_labels = batch_text_labels.cuda()
            with torch.no_grad():
                val_loss, val_pred_logits = model(batch_imgs, batch_text_labels, batch_rectangles, batch_text_labels_mask)

            this_time = time.time() - start_time
            log_info = "val iter :{}, time: {:.2f} Loss: {:.3f}".format(i, this_time, val_loss.data)
            log_file.write(log_info + '\n')
            print(log_info)

        if (i + 1) % args.save_iter == 0:
            torch.save(
                model.state_dict(), args.save_name + '_{}.pth'.format(i + 1)
            )
        if i > 0 and i % args.lr_step == 0:                # 调整学习速率
            lrScheduler.step()
        # torch.cuda.empty_cache()
    print('end the training')
    log_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a recognizer")
    parser.add_argument('config', help='config file')
    # parser.add_argument('--')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    train(cfg)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()
