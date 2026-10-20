# -*- coding: utf-8 -*-
"""决策规则对比：阈值 T + 最小间隔 margin（centroid 聚合）"""
import os

import numpy as np

from engine import FaceEngine

GAL = os.getenv("GALLERY_DIR", "/data/gallery")
eng = FaceEngine()

names, vecs = [], []
for name in sorted(os.listdir(GAL)):
    d = os.path.join(GAL, name)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        with open(os.path.join(d, fn), "rb") as f:
            bgr, _r, faces = eng.detect_all(f.read())
        if len(faces) == 0:
            continue
        names.append(name)
        vecs.append(eng.embed(bgr, eng.largest_face(faces)))

names = np.array(names)
M = np.stack(vecs)
print("样本数:", len(names))


def scores_for(i):
    """返回 {人名: 相似度}（用除自身外的样本算类中心）"""
    out = {}
    for nm in np.unique(names):
        mask = (names == nm) & (np.arange(len(M)) != i)
        if not mask.any():
            continue
        c = M[mask].mean(axis=0)
        out[nm] = float((c / np.linalg.norm(c)) @ M[i])
    return out


rows = []
for i in range(len(M)):
    s = scores_for(i)
    order = sorted(s.items(), key=lambda kv: -kv[1])
    best_nm, best = order[0]
    second = order[1][1] if len(order) > 1 else -1
    rows.append((names[i], best_nm, best, second))

best_combo, best_acc = None, -1
for t in np.arange(0.25, 0.75, 0.01):
    for mg in np.arange(0.0, 0.35, 0.01):
        ok = 0
        for truth, pred, best, second in rows:
            accept = best >= t and (best - second) >= mg
            if (accept and pred == truth) or (not accept and pred != truth):
                ok += 1
        acc = ok / len(rows)
        if acc > best_acc:
            best_acc, best_combo = acc, (t, mg)

t, mg = best_combo
print()
print("仅用阈值：最佳 T=0.440 准确率约 83%%（上一轮结果）")
print("阈值+间隔：最佳 T=%.2f  margin=%.2f  准确率=%.1f%%" % (t, mg, best_acc * 100))
print()
print("在该组合下逐条结果：")
for truth, pred, best, second in rows:
    accept = best >= t and (best - second) >= mg
    flag = "OK  " if ((accept and pred == truth) or (not accept and pred != truth)) else "WRONG"
    print("  %-6s 真实=%-6s 预测=%-6s best=%.3f second=%.3f 判定=%s  %s"
          % ("", truth, pred, best, second, "接受:" + pred if accept else "拒绝(未知)", flag))
