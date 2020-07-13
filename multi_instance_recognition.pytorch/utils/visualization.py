import cv2
import numpy as np
import math
import os

def mask_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])

    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break
        att_map = cv2.resize(att_map, (W, H))
        _att_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])
        _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)

        show_attention = cv2.addWeighted(img, 0.5, _att_map, 2, 0)
        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), show_attention)

    return True

def contour_visualize(img, alpha, pred, vis_dir, img_path, num_line=10, equal_bound=1e-4):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])

    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break

        att_map = cv2.resize(att_map, (W, H))
        max_weights = np.max(att_map)
        min_weights = np.max(att_map)

        weights_rank = np.arange(min_weights, max_weights, num_line, dtype=np.float32)

        pass


def estimate_gauss(alpha): # N * T * H * W
    H, W = alpha.shape[2], alpha.shape[3]
    alpha = alpha.reshape([-1, H, W])
    # alpha[np.where(alpha<=0.1)] = 0.
    N = alpha.shape[0]
    alpha_x = np.sum(alpha, axis=1) # N * W
    alpha_y = np.sum(alpha, axis=2) # N * H

    coord_x = np.arange(0, W).astype(np.float32) # W
    coord_y = np.arange(0, H).astype(np.float32) # H

    mu_x = np.matmul(alpha_x, np.expand_dims(coord_x, axis=1))  # N * 1
    mu_y = np.matmul(alpha_y, np.expand_dims(coord_y, axis=1))  # N * 1

    sigma_x = np.matmul(alpha_x, np.expand_dims(np.power(coord_x, 2), axis=1)) - np.power(mu_x, 2)  # N * 1
    sigma_y = np.matmul(alpha_y, np.expand_dims(np.power(coord_y, 2), axis=1)) - np.power(mu_y, 2)  # N * 1
    mu_x, mu_y = np.tile(mu_x, [1, W]), np.tile(mu_y, [1, H])  # N * W & N * H
    sigma_x, sigma_y = np.tile(sigma_x, [1, W]), np.tile(sigma_y, [1, H])  # N * W & N * H
    coord_x, coord_y = np.tile(np.expand_dims(coord_x, axis=0), [N, 1]), np.tile(np.expand_dims(coord_y, axis=0), [N, 1])

    gauss_x = np.exp(-1. * np.power(coord_x - mu_x, 2) / (2. * sigma_x))
    gauss_y = np.exp(-1. * np.power(coord_y - mu_y, 2) / (2. * sigma_y))

    gauss_x[np.where(gauss_x <= math.exp(-0.5))] = 0.
    gauss_y[np.where(gauss_y <= math.exp(-0.5))] = 0.

    coefficient_x = 1. / (np.sqrt(2. * math.pi * sigma_x))  # N * W
    coefficient_y = 1. / (np.sqrt(2. * math.pi * sigma_y))  # N * H

    gauss_x = gauss_x * coefficient_x
    gauss_y = gauss_y * coefficient_y


    gauss_x = gauss_x / np.expand_dims(np.sum(gauss_x, axis=1), axis=1)
    gauss_y = gauss_y / np.expand_dims(np.sum(gauss_y, axis=1), axis=1)

    # gauss_2d = np.matmul(np.expand_dims(gauss_y, axis=2), np.expand_dims(gauss_x, axis=1))  # N * H * W

    return gauss_x, gauss_y

def line_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha_H, alpha_W = alpha.shape[2], alpha.shape[3]
    # alpha = estimate_gauss(alpha)
    alpha = alpha.reshape([-1, (alpha_H * alpha_W)])
    # alpha = alpha / np.expand_dims(np.sum(alpha, axis=1), axis=1)
    alpha = alpha.reshape([-1, alpha_H, alpha_W, 1])

    for i, att_map in enumerate(alpha): # H * W * 1
        img_line = img.copy()
        if i >= len(pred):
            break
        # att_map = cv2.resize(att_map, (W, H)) # H * W * 1
        x_axis_att_map = np.sum(att_map, axis=0).reshape(alpha_W) # W
        y_axis_att_map = np.sum(att_map, axis=1).reshape(alpha_H) # H

        key_coord_x = np.arange(0, W, W//alpha_W)
        key_coord_y = np.arange(0, H, H//alpha_H)
        coord_x = np.arange(0, W)
        coord_y = np.arange(0, H)
        x_axis_att_value = np.interp(coord_x, key_coord_x, x_axis_att_map)  # 100
        x_axis_att_value = x_axis_att_value * H
        y_axis_att_value = np.interp(coord_y, key_coord_y, y_axis_att_map)  # 100
        y_axis_att_value = y_axis_att_value * H


        x_att_pts = np.stack([coord_x, x_axis_att_value]).transpose((1, 0)).astype(np.int32)
        x_att_pts[:, 1] = H - x_att_pts[:, 1]

        y_att_pts = np.stack([y_axis_att_value, coord_y]).transpose((1, 0)).astype(np.int32)

        img_line = cv2.polylines(img_line, [x_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)
        img_line = cv2.polylines(img_line, [y_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)

        att_map = cv2.resize(att_map, (W, H))
        _att_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])
        _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)

        mask_line_img = cv2.addWeighted(img_line, 0.5, _att_map, 2, 0)

        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), mask_line_img)

    return True


def line_mask_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha_H, alpha_W = alpha.shape[2], alpha.shape[3]
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])

    for i, att_map in enumerate(alpha):  # H * W * 1
        img_line = img.copy()
        if i >= len(pred):
            break
        # att_map = cv2.resize(att_map, (W, H)) # H * W * 1
        x_axis_att_map = np.sum(att_map, axis=0).reshape(alpha_W)  # W
        y_axis_att_map = np.sum(att_map, axis=1).reshape(alpha_H)  # H

        key_coord_x = np.arange(0, W, W // alpha_W)
        key_coord_y = np.arange(0, H, H // alpha_H)
        coord_x = np.arange(0, W)
        coord_y = np.arange(0, H)
        x_axis_att_value = np.interp(coord_x, key_coord_x, x_axis_att_map)  # 100
        x_axis_att_value = x_axis_att_value * H
        y_axis_att_value = np.interp(coord_y, key_coord_y, y_axis_att_map)  # 100
        y_axis_att_value = y_axis_att_value * H

        x_att_pts = np.stack([coord_x, x_axis_att_value]).transpose((1, 0)).astype(np.int32)
        x_att_pts[:, 1] = H - x_att_pts[:, 1]

        y_att_pts = np.stack([y_axis_att_value, coord_y]).transpose((1, 0)).astype(np.int32)

        img_line = cv2.polylines(img_line, [x_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)
        img_line = cv2.polylines(img_line, [y_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])),
                    img_line)

    return True

def heatmap_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    if len(alpha.shape) == 4:
        alpha = np.expand_dims(alpha, axis=2)
    H, W, _ = img.shape
    # alpha: 1 * T * h * H * W
    alpha = alpha.reshape([alpha.shape[1], alpha.shape[2], alpha.shape[3], alpha.shape[4], 1])
    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break
        for j, h_att_map in enumerate(att_map): # H * W * 1
            h_att_map = cv2.resize(h_att_map, (W, H))
            att_max = h_att_map.max()
            h_att_map /= att_max
            h_att_map *= 255
            h_att_map = h_att_map.astype(np.uint8)
            heatmap = cv2.applyColorMap(h_att_map, cv2.COLORMAP_JET)

            show_attention = img.copy()
            show_attention = cv2.addWeighted(heatmap, 0.5, show_attention, 0.5, 0,dtype=cv2.CV_32F)
            cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, j, pred[i])), show_attention)

    return True