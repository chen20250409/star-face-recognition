# -*- coding: utf-8 -*-
"""阈值校准：留一法（leave-one-out）跑一遍全部底库样本

对每个样本，把它自己从底库里剔除后再比对，得到：
  - correct 分数：与「同一个人其它照片」的最高相似度（模拟"同一人的新照片"）
  - impostor 分数：与「其它人」的最高相似度（模拟"陌生人"）
据此给出推荐阈值。
"""
import numpy as np

from database import FaceSample, Person, SessionLocal, init_db, wait_for_db

if not wait_for_db():
    raise SystemExit("连不上 MySQL")
init_db()

with SessionLocal() as s:
    persons = {p.id: p.name for p in s.query(Person).all()}
    rows = s.query(FaceSample).all()

if not rows:
    raise SystemExit("底库为空")

vecs = np.stack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows])
owners = np.array([r.person_id for r in rows])
names = [persons[o] for o in owners]
print("底库：%d 个样本 / %d 人" % (len(vecs), len(persons)))

correct, impostor = [], []
detail = {}
for i in range(len(vecs)):
    sims = vecs @ vecs[i]
    sims[i] = -1.0                      # 剔除自身
    per_person = {}
    for pid in np.unique(owners):
        mask = (owners == pid) & (np.arange(len(vecs)) != i)
        if not mask.any():
            continue
        # 与线上一致：用该人其余样本的类中心做比对
        c = vecs[mask].mean(axis=0)
        n = float(np.linalg.norm(c))
        per_person[pid] = float((c / n if n > 0 else c) @ vecs[i])
    mine = per_person.get(owners[i], -1.0)
    others = max([v for k, v in per_person.items() if k != owners[i]], default=-1.0)
    correct.append(mine)
    impostor.append(others)
    detail.setdefault(names[i], {"correct": [], "impostor": []})
    detail[names[i]]["correct"].append(mine)
    detail[names[i]]["impostor"].append(others)

correct, impostor = np.array(correct), np.array(impostor)
print()
print("%-10s %6s %6s %6s   %6s %6s" % ("人名", "最低", "平均", "最高", "同人最差", "异人最高"))
for n, d in detail.items():
    c = np.array(d["correct"])
    im = np.array(d["impostor"])
    print("%-10s %6.3f %6.3f %6.3f   %6.3f %6.3f"
          % (n, c.min(), c.mean(), c.max(), c.min(), im.max()))

print()
print("全体 correct  分数：min=%.3f  p05=%.3f  中位=%.3f  max=%.3f"
      % (correct.min(), np.percentile(correct, 5), np.median(correct), correct.max()))
print("全体 impostor 分数：min=%.3f  中位=%.3f  p95=%.3f  max=%.3f"
      % (impostor.min(), np.median(impostor), np.percentile(impostor, 95), impostor.max()))

best_t, best_acc = 0.0, -1
for t in np.arange(0.10, 0.80, 0.005):
    acc = ((correct >= t).sum() + (impostor < t).sum()) / (len(correct) + len(impostor))
    if acc > best_acc:
        best_acc, best_t = acc, float(t)

print()
print("最佳阈值 = %.3f  （准确率 %.1f%%，正样本召回 %d/%d，陌生人拒绝 %d/%d）"
      % (best_t, best_acc * 100, (correct >= best_t).sum(), len(correct),
         (impostor < best_t).sum(), len(impostor)))
if correct.min() > impostor.max():
    print("两类分数完全可分：correct 最低 %.3f > impostor 最高 %.3f，中间任取都安全"
          % (correct.min(), impostor.max()))
    print("推荐取中点 %.3f" % ((correct.min() + impostor.max()) / 2))
else:
    print("注意：两类分数有重叠区间 [%.3f, %.3f]，阈值只能在误拒/误受之间取平衡"
          % (impostor.max(), correct.min()))
