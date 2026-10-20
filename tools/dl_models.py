# -*- coding: utf-8 -*-
"""在服务器上直接下载两个 ONNX 模型并校验"""
import hashlib
import os
import time
import urllib.request

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
TARGETS = {
    "face_detection_yunet_2023mar.onnx": (f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx", 232589),
    "face_recognition_sface_2021dec.onnx": (f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx", 38696353),
}
OUT = os.path.expanduser("~/face-app/models")

os.makedirs(OUT, exist_ok=True)

for name, (url, expect) in TARGETS.items():
    path = os.path.join(OUT, name)
    if os.path.exists(path) and os.path.getsize(path) == expect:
        print("SKIP %s 已存在且大小正确" % name, flush=True)
        continue
    t = time.time()
    last = 0

    def hook(b, bs, total, _t=t, _last=[0]):
        done = b * bs
        if done - _last[0] > 4 * 1024 * 1024:
            _last[0] = done
            print("  %s %d/%d  %.0fs" % (name, done, total, time.time() - _t), flush=True)

    print("DOWNLOAD %s" % name, flush=True)
    urllib.request.urlretrieve(url, path, hook)
    size = os.path.getsize(path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    lfs = b"git-lfs" in open(path, "rb").read(200)
    print("%s size=%d expect=%d %s sha256=%s lfs_pointer=%s  %.0fs"
          % (name, size, expect, "OK" if size == expect else "MISMATCH", h, lfs, time.time() - t), flush=True)

print("ALL_DONE", flush=True)
