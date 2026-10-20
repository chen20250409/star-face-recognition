# -*- coding: utf-8 -*-
"""方案对比实验：聚合方式 × 是否水平翻转 TTA

输出每种组合的 correct / impostor 分离度与最佳阈值下的准确率。
"""
import os

import cv2
import numpy as np

from engine import FaceEngine

GAL = os.getenv("GALLERY_DIR", "/data/gallery")
eng = FaceEngine()


def embed_with_tta(path):
    with open(path, "rb") as f:
        data = f.read()
    bgr, _rgb, faces = eng.detect_all(data)
    if len(faces) == 0:
        return None
    row = eng.largest_face(faces)
    v1 = eng.embed(bgr, row)
    v2 = eng.embed(cv2.flip(bgr, 1), row)
    v = v1 + v2
    return (v / np.linalg.norm(v)).astype(np.float32)


names, vecs_tta, vecs_raw = [], [], []
for name in sorted(os.listdir(GAL)):
    d = os.path.join(GAL, name)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        p = os.path.join(d, fn)
        v = embed_with_tta(p)
        if v is None:
            print("跳过（未检出人脸）", name, fn)
            continue
        names.append(name)
        vecs_tta.append(v)
        with open(p, "rb") as f:
            bgr, _rgb, faces = eng.detect_all(f.read())
        vecs_raw.append(eng.embed(bgr, eng.largest_face(faces)))

names = np.array(names)
M_tta = np.stack(vecs_tta)
M_raw = np.stack(vecs_raw)
print("样本数:", len(names))


def evaluate(M, tag, agg):
    correct, impostor = [], []
    for i in range(len(M)):
        sims = M @ M[i]
        sims[i] = -1
        per = {}
        for nm in np.unique(names):
            mask = (names == nm) & (np.arange(len(M)) != i)
            if not mask.any():
                continue
            s = np.sort(sims[mask])[::-1]
            if agg == "max":
                per[nm] = float(s[0])
            elif agg == "top3":
                per[nm] = float(s[:min(3, len(s))].mean())
            elif agg == "centroid":
                c = M[mask].mean(axis=0)
                c = c / np.linalg.norm(c)
                per[nm] = float(c @ M[i])
        correct.append(per[names[i]])
        impostor.append(max([v for k, v in per.items() if k != names[i]], default=-1))
    correct, impostor = np.array(correct), np.array(impostor)
    best_t, best_acc = 0, -1
    for t in np.arange(0.05, 0.90, 0.005):
        acc = ((correct >= t).sum() + (impostor < t).sum()) / (2 * len(correct))
        if acc > best_acc:
            best_acc, best_t = acc, t
    overlap = correct.min() <= impostor.max()
    print("%-28s correct[%.3f~%.3f] impostor[%.3f~%.3f]  最佳阈值=%.3f 准确率=%.1f%% %s"
          % (tag, correct.min(), correct.max(), impostor.min(), impostor.max(),
             best_t, best_acc * 100, "有重叠" if overlap else "完全可分"))
    return best_acc


print()
print("=== 原始特征（无 TTA）")
evaluate(M_raw, "raw + max", "max")
evaluate(M_raw, "raw + top3mean", "top3")
evaluate(M_raw, "raw + centroid", "centroid")
print()
print("=== 水平翻转 TTA（原图与镜像特征取平均）")
evaluate(M_tta, "tta + max", "max")
evaluate(M_tta, "tta + top3mean", "top3")
evaluate(M_tta, "tta + centroid", "centroid")
