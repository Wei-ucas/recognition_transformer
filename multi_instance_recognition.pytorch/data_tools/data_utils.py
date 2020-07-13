import string
import numpy as np
import cv2
import math
import itertools
import random

def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):

    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.

def check_and_validate_polys(polys, tags, labels, label_masks, words, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme

    validated_polys = []
    validated_tags = []
    validated_labels = []
    validated_label_masks = []
    validated_words = []

    if polys.shape[0] == 0:
        # return polys
        return np.array(validated_polys), np.array(validated_tags), validated_labels
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    for poly, tag, label, mask, word in zip(polys, tags, labels, label_masks, words):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            # print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]

        # if not check_is_horizon(poly):
        # print("vertical text area")
        # continue
        validated_polys.append(poly)
        validated_tags.append(tag)
        validated_labels.append(label)
        validated_label_masks.append(mask)
        validated_words.append(word)

    return np.array(validated_polys), np.array(validated_tags), np.array(validated_labels), np.array(validated_label_masks), validated_words

def rotate_image(img, boxes, angle, scale=1):
    H, W, _ = img.shape
    rangle = np.deg2rad(angle)  # angle in radians
    new_width = (abs(np.sin(rangle) * H) + abs(np.cos(rangle) * W)) * scale
    new_height = (abs(np.cos(rangle)*H) + abs(np.sin(rangle)*W))*scale

    rot_mat = cv2.getRotationMatrix2D((new_width * 0.5, new_height * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_width - W) * 0.5, (new_height - H) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(new_width)), int(math.ceil(new_height))), flags=cv2.INTER_LANCZOS4)

    rot_bboxes = list()
    for bbox in boxes:
        new_box = []
        for point in bbox:
            r_point = np.dot(rot_mat, np.array([point[0], point[1], 1]))
            new_box.append(r_point)
        rot_bboxes.append(new_box)
        # point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
    return rot_img, np.array(rot_bboxes)

def crop_box(polys, num=4, iratio=0.5, israndom=True):
    """
    :param polys: N * 4 * 2
    :param num:
    :return:
    """
    boxes = []
    for poly in polys: # 4 * 2
        instances = []
        min_x, max_x = np.min(poly[:, 0]), np.max(poly[:, 0])
        min_y, max_y = np.min(poly[:, 1]), np.max(poly[:, 1])
        # instances.append([1, min_x, min_y, max_x, max_y])
        h = max_y - min_y
        w = max_x - min_x
        ins_id = np.arange(0,num)
        np.random.shuffle(ins_id)
        for i in ins_id:
        # for i in range(num):
            min_x, max_x = np.min(poly[:, 0]), np.max(poly[:, 0])
            min_y, max_y = np.min(poly[:, 1]), np.max(poly[:, 1])
            if israndom:
                ratio = random.random() * iratio
            else:
                ratio = iratio
            # ratio = 0.3
            if i % 4 == 0:
                min_x += w * ratio
            elif i % 4 == 1:
                max_x -= w * ratio
            elif i % 4 == 2:
                min_y += h * ratio
            else:
                max_y -= h * ratio
            instances.append([1, min_x, min_y, max_x, max_y])
        boxes.append(instances)
    return boxes

