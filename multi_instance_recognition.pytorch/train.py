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
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.distributed as dist
from data_tools.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
import logging
from model import MultiInstanceRecognition
# from data_tools.instance_set import ImgInstanceLoader
from data_tools.concat_dataset import build_dataloader
from utils.logging import setup_logger
from utils.dist_reduce import reduce_loss_dict, get_rank

torch.backends.cudnn.enabled = False


def train(cfg, args):
    logger = logging.getLogger('model training')
    train_data = build_dataloader(cfg.train_data_cfg, args.distributed)
    logger.info("train data: {}".format(len(train_data)))
    val_data = build_dataloader(cfg.val_data_cfg, args.distributed)
    logger.info("val data: {}".format(len(val_data)))

    model = MultiInstanceRecognition(cfg.model_cfg).cuda()
    if cfg.resume_from is not None:
        logger.info('loading pretrained models from {opt.continue_model}')
        model.load_state_dict(torch.load(cfg.resume_from))
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    voc, char2id, id2char = get_vocabulary("ALLCASES_SYMBOLS")

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logger.info('Trainable params num : ', sum(params_num))
    optimizer = optim.Adam(filtered_parameters, lr=cfg.lr, betas=(0.9, 0.999))
    lrScheduler = lr_scheduler.MultiStepLR(optimizer, [1, 2, 3], gamma=0.1)

    max_iters = cfg.max_iters
    start_iter = 0
    if cfg.resume_from is not None:
        start_iter = int(cfg.resume_from.split('_')[-1].split('.')[0])
        logger.info('continue to train, start_iter: {start_iter}')

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

        batch_imgs = batch_imgs.cuda(non_blocking=True)
        batch_rectangles = batch_rectangles.cuda(non_blocking=True)
        batch_text_labels = batch_text_labels.cuda(non_blocking=True)
        data_time = time.time() - data_time_s
        # print(time.time() -s)
        # s = time.time()
        loss, decoder_logits = model(batch_imgs, batch_text_labels, batch_rectangles, batch_text_labels_mask)
        del batch_data
        # print(time.time() - s)
        # print('------')
        # s = time.time()

        loss = loss.mean()
        print(loss)
        # del loss
        # print(time.time() - s)
        # print('------')

        if i % cfg.train_verbose == 0:
            this_time = time.time() - start_time
            if args.distributed:
                loss = dist.reduce(loss,0)
            log_info = "train iter :{}, time: {:.2f}, data_time: {:.2f}, Loss: {:.3f}".format(i, this_time, data_time, loss.data)
            logger.info(log_info)
            torch.cuda.empty_cache()
            # break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss
        if i % cfg.val_iter == 0:
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
            del val_batch
            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_rectangles = batch_rectangles.cuda(non_blocking=True)
            batch_text_labels = batch_text_labels.cuda(non_blocking=True)
            with torch.no_grad():
                val_loss, val_pred_logits = model(batch_imgs, batch_text_labels, batch_rectangles, batch_text_labels_mask)
            pred_labels = val_pred_logits.argmax(dim=2).cpu().numpy()
            pred_value_str = idx2label(pred_labels, id2char, char2id)
            # gt_str = batch_words
            gt_str = []
            for words in batch_words:
                gt_str = gt_str + words
            val_dec_metrics_result = calc_metrics(pred_value_str,
                                                  gt_str, metrics_type="accuracy")
            this_time = time.time() - start_time
            if args.distributed:
                loss = dist.reduce(val_loss,0)
            log_info = "val iter :{}, time: {:.2f} Loss: {:.3f}, acc: {:.2f}".format(i, this_time, loss.mean().data, val_dec_metrics_result)
            logger.info(log_info)
            del val_loss
        if (i + 1) % cfg.save_iter == 0:
            torch.save(
                model.state_dict(), cfg.save_name + '_{}.pth'.format(i + 1)
            )
        if i > 0 and i % cfg.lr_step == 0:                # 调整学习速率
            lrScheduler.step()
            logger.info("lr step")
        # torch.cuda.empty_cache()
    print('end the training')


def parse_args():
    parser = argparse.ArgumentParser(description="Train a recognizer")
    parser.add_argument('config', help='config file')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    # parser.add_argument('--')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    local_rank = args.local_rank

    logger = setup_logger(__name__, cfg.save_name, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config))
    logger.info(cfg._text)

    train(cfg, args)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()
