# -*- coding: utf-8 -*-
"""在服务器上直接下载模型并校验

- 人脸：YuNet(检测) + SFace(识别)，来自 opencv_zoo（Git LFS，走 media 直链）
- 物品：YOLOv4-tiny(cfg + weights) + coco.names，来自 AlexeyAB/darknet
"""
import hashlib
import os
import time
import urllib.request

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
DARKNET = "https://raw.githubusercontent.com/AlexeyAB/darknet/master"

TARGETS = {
    # 文件名: (url, 期望字节数, 说明)
    "face_detection_yunet_2023mar.onnx": (
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx", 232589, "YuNet 人脸检测"),
    "face_recognition_sface_2021dec.onnx": (
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx", 38696353, "SFace 人脸识别"),
    "yolov4-tiny.cfg": (f"{DARKNET}/cfg/yolov4-tiny.cfg", 3231, "YOLOv4-tiny 配置"),
    "yolov4-tiny.weights": (
        "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
        24251276, "YOLOv4-tiny 权重"),
    "coco.names": (f"{DARKNET}/data/coco.names", 625, "COCO 80 类名"),
}

OUT = os.path.expanduser("~/face-app/models")
os.makedirs(OUT, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for name, (url, expect, desc) in TARGETS.items():
    path = os.path.join(OUT, name)
    if os.path.exists(path) and os.path.getsize(path) == expect:
        print("[跳过] %-38s %s 已存在且大小正确" % (name, desc), flush=True)
        continue
    t = time.time()
    print("[下载] %-38s %s" % (name, desc), flush=True)
    last = [0]

    def hook(b, bs, total, _t=t, _last=last, _n=name):
        done = b * bs
        if done - _last[0] > 8 * 1024 * 1024:
            _last[0] = done
            print("    %s %d/%d  %.0fs" % (_n, done, total, time.time() - _t), flush=True)

    urllib.request.urlretrieve(url, path, hook)
    size = os.path.getsize(path)
    ok = (size == expect)
    print("[完成] %-38s size=%d expect=%d %s  sha256=%s  %.0fs"
          % (name, size, expect, "OK" if ok else "大小不符!", sha256(path)[:16], time.time() - t),
          flush=True)

print("ALL_DONE", flush=True)
