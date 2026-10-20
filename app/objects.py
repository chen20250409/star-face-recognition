# -*- coding: utf-8 -*-
"""物品检测：OpenCV DNN 跑 YOLOv4-tiny（COCO 80 类）

不引入 torch / ultralytics，只用 cv2.dnn，模型 24MB，CPU 单帧约 80–150ms。
"""
import logging
import os

import cv2
import numpy as np

log = logging.getLogger("objects")

OBJ_CFG = os.getenv("OBJ_CFG", "/models/yolov4-tiny.cfg")
OBJ_WEIGHTS = os.getenv("OBJ_WEIGHTS", "/models/yolov4-tiny.weights")
OBJ_NAMES = os.getenv("OBJ_NAMES", "/models/coco.names")
OBJ_SCORE = float(os.getenv("OBJ_SCORE", "0.25"))
OBJ_NMS = float(os.getenv("OBJ_NMS", "0.40"))
OBJ_SIZE = int(os.getenv("OBJ_SIZE", "416"))
OBJ_MAX = int(os.getenv("OBJ_MAX", "12"))
# 人脸框落在 person 框内的比例超过该值，就认为这个 person 框是"已识别的这个人"，不再重复标注
FACE_COVER = float(os.getenv("OBJ_FACE_COVER", "0.85"))

# COCO 80 类的中文名，顺序必须与 coco.names 一致
CN_NAMES = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船", "红绿灯",
    "消防栓", "停车标志", "停车计时器", "长椅", "鸟", "猫", "狗", "马", "羊", "牛",
    "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘",
    "滑雪板", "单板滑雪", "运动球", "风筝", "棒球棍", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
    "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子",
    "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽", "床",
    "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
    "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷",
]


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _cover(inner, outer):
    """inner 有多少比例的面积落在 outer 里。

    实测（27 张底库图）：YOLO 的 person 框是全身/半身框，和人脸框的 IoU 只有
    0.04~0.09，但人脸框 100% 被它包住、面积比 11~26 倍。所以判断"这个 person 框
    就是已识别的这个人"必须用包含比例，不能用 IoU。
    """
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (aw * ah) if aw * ah > 0 else 0.0


class ObjectDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(OBJ_CFG, OBJ_WEIGHTS)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_layers = self.net.getUnconnectedOutLayersNames()
        with open(OBJ_NAMES, encoding="utf-8") as f:
            self.names = [l.strip() for l in f if l.strip()]
        log.info("物品检测模型已加载：%d 类 输入=%d conf=%.2f nms=%.2f",
                 len(self.names), OBJ_SIZE, OBJ_SCORE, OBJ_NMS)

    def detect(self, bgr, face_boxes=None):
        """返回 [{label, label_cn, score, bbox}]，已按分数降序并做 NMS 去重"""
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1.0 / 255.0, (OBJ_SIZE, OBJ_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)

        boxes, scores, ids = [], [], []
        for out in outs:
            for det in out:
                cls_scores = det[5:]
                cid = int(np.argmax(cls_scores))
                conf = float(cls_scores[cid]) * float(det[4])
                if conf < OBJ_SCORE:
                    continue
                cx, cy, bw, bh = det[0] * w, det[1] * h, det[2] * w, det[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                scores.append(conf)
                ids.append(cid)

        if not boxes:
            return []

        idx = cv2.dnn.NMSBoxes(boxes, scores, OBJ_SCORE, OBJ_NMS)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()

        result = []
        face_boxes = face_boxes or []
        for i in idx:
            cid = ids[i]
            label = self.names[cid] if cid < len(self.names) else str(cid)
            box = boxes[i]
            # 已经识别人脸的这个人，不再重复画 person 框（否则大框套小框很难看）
            # 没有检出人脸的 person 框保留，比如路人、背景里的人
            if label == "person" and any(_cover(fb, box) >= FACE_COVER for fb in face_boxes):
                continue
            result.append({
                "label": label,
                "label_cn": CN_NAMES[cid] if cid < len(CN_NAMES) else label,
                "score": round(float(scores[i]), 4),
                "bbox": [float(v) for v in box],
            })

        result.sort(key=lambda r: -r["score"])
        return result[:OBJ_MAX]
