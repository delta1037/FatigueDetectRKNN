import torch
import numpy as np
from PIL import Image


def cut_resize_letterbox(image, det, target_size):
    iw, ih = image.size

    facebox_x = int(det[0])
    facebox_y = int(det[1])
    facebox_w = int(det[2])
    facebox_h = int(det[3])

    facebox_max_length = max(facebox_w, facebox_h)
    width_margin_length = int((facebox_max_length - facebox_w) / 2)
    height_margin_length = int((facebox_max_length - facebox_h) / 2)

    face_letterbox_x = facebox_x - width_margin_length
    face_letterbox_y = facebox_y - height_margin_length
    face_letterbox_w = facebox_max_length
    face_letterbox_h = facebox_max_length

    top = -face_letterbox_y if face_letterbox_y < 0 else 0
    left = -face_letterbox_x if face_letterbox_x < 0 else 0
    bottom = face_letterbox_y + face_letterbox_h - ih if face_letterbox_y + face_letterbox_h - ih > 0 else 0
    right = face_letterbox_x + face_letterbox_w - iw if face_letterbox_x + face_letterbox_w - iw > 0 else 0

    margin_image = Image.new('RGB', (iw + right - left, ih + bottom - top), (0, 0, 0))
    margin_image.paste(image, (left, top))

    face_letterbox = margin_image.crop(
        (face_letterbox_x, face_letterbox_y, face_letterbox_x + face_letterbox_w, face_letterbox_y + face_letterbox_h))
    face_letterbox = face_letterbox.resize(target_size, Image.Resampling.BICUBIC)

    return face_letterbox, facebox_max_length / target_size[0], face_letterbox_x, face_letterbox_y


def pad_image(image, target_size):
    iw, ih = image.size
    w, h = target_size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale + 0.5)
    nh = int(ih * scale + 0.5)

    pad_w = (w - nw) // 2
    pad_h = (h - nh) // 2
    image = image.resize((nw, nh), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))

    new_image.paste(image, (pad_w, pad_h))

    return new_image, scale, pad_w, pad_h


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_img_tensor(pil_img, use_cuda, target_size, transform):
    iw, ih = pil_img.size
    if iw != target_size[0] or ih != target_size[1]:
        pil_img = pil_img.resize(target_size, Image.Resampling.BICUBIC)

    tensor_img = transform(pil_img)
    tensor_img = torch.unsqueeze(tensor_img, 0)
    if use_cuda:
        tensor_img = tensor_img.cuda()

    return tensor_img


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(confidences, boxes, prob_threshold=0.7, image_size=(320, 240), iou_threshold=0.3, top_k=1):
    boxes = np.squeeze(boxes)
    confidences = np.squeeze(confidences)
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= image_size[0]
    picked_box_probs[:, 1] *= image_size[1]
    picked_box_probs[:, 2] *= image_size[0]
    picked_box_probs[:, 3] *= image_size[1]
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


# 获取大于 prob_threshold 的最优的结果
def get_predict_box(confidences, boxes, image_size=(320, 240), prob_threshold=0.7):
    boxes = np.squeeze(boxes)
    confidences = np.squeeze(confidences)
    # 过滤结果
    probs = confidences[:, 1]
    mask = probs > prob_threshold
    probs = probs[mask]
    if probs.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    boxes = boxes[mask, :]

    max_idx = 0
    max_prob = probs[0]
    for idx in range(len(probs)):
        if probs[idx] > max_prob:
            max_idx = idx
            max_prob = probs[idx]
    boxes[max_idx, 0] *= image_size[0]
    boxes[max_idx, 1] *= image_size[1]
    boxes[max_idx, 2] *= image_size[0]
    boxes[max_idx, 3] *= image_size[1]
    return np.array([boxes[max_idx]]), np.array([1]), np.array([max_prob])
