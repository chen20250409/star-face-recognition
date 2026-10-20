# -*- coding: utf-8 -*-
"""脏数据实验二：整体偷换标签

把【景甜】的 9 张全部以【鞠婧祎】的名义注册，再把「景甜」这个人整个删掉，
然后用景甜的照片去识别 —— 看系统会不会照单全收。

做法前先把图片读进内存，因为删除人物会连底库目录一起清掉。
"""
import io
import json
import os
import uuid
import urllib.error
import urllib.request

BASE = os.getenv("BASE", "http://127.0.0.1")
GALLERY = os.getenv("GALLERY_DIR", "/data/gallery")
A, B = "景甜", "鞠婧祎"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())


def post_multipart(path, files, fields=None):
    b = "----exp" + uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in (fields or {}).items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode())
    for field, filename, data in files:
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                    "Content-Type: image/jpeg\r\n\r\n" % (b, field, filename)).encode())
        body.write(data)
        body.write(b"\r\n")
    body.write(("--%s--\r\n" % b).encode())
    req = urllib.request.Request(BASE + path, data=body.getvalue(), method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + b)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}")


def delete(path):
    req = urllib.request.Request(BASE + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}")


def pids():
    return {p["name"]: p["id"] for p in get("/api/persons")["persons"]}


def photos(pid):
    return get("/api/persons/%d/photos" % pid)["photos"]


def recog(data, fn):
    j = post_multipart("/api/recognize", [("files", fn, data)], {"detect_objects": "false"})
    res = j["results"][0]
    if not res["faces"]:
        return "(未检出人脸)", 0.0, 0.0
    f = res["faces"][0]
    return f["name"], f["score"], f["margin"]


d = pids()
print("=" * 74)
print("初始：", {k: len(photos(v)) for k, v in d.items()})
print("=" * 74)

jing_dir = os.path.join(GALLERY, A)
names = sorted(n for n in os.listdir(jing_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))
blobs = [(n, open(os.path.join(jing_dir, n), "rb").read()) for n in names]
query = blobs[:5]
print("  景甜 %d 张已读入内存（删除人物会同时删目录，必须先读）" % len(blobs))

# 鞠婧祎原本的 9 张也先读进内存：注入之后目录里会混进景甜的图，
# 再按文件名排序取「前几张」会取错人（实测踩过这个坑）
orig_ju = []
for p in photos(d[B]):
    fp = os.path.join(GALLERY, p["file"])
    if os.path.isfile(fp):
        orig_ju.append((os.path.basename(p["file"]), open(fp, "rb").read()))
print("  鞠婧祎原图 %d 张已读入内存（按 sid 精确对应，不看文件名顺序）" % len(orig_ju))

print()
print("=" * 74)
print("步骤 1 · 把景甜的 %d 张全部以「鞠婧祎」的名义注册" % len(blobs))
print("=" * 74)
r = post_multipart("/api/persons", [("files", n, dt) for n, dt in blobs], {"name": B})
print("  ", r)
print("  鞠婧祎现在特征数：%d" % len(photos(d[B])))

print()
print("=" * 74)
print("步骤 2 · 删除「景甜」这个人物（模拟标签被整体偷换）")
print("=" * 74)
print("  ", delete("/api/persons/%d" % d[A]))
print("  剩余人物：", {k: len(photos(v)) for k, v in pids().items()})

print()
print("=" * 74)
print("步骤 3 · 拿景甜的照片去识别，此时底库里已经没有「景甜」了")
print("=" * 74)
print("    %-26s %-10s %8s %8s" % ("景甜的照片", "判成", "相似度", "margin"))
rows = [(n,) + recog(dt, n) for n, dt in query]
for fn, name, s, m in rows:
    print("    %-26s %-10s %8.4f %8.4f" % (fn[:26], name, s, m))
print("    → 认成鞠婧祎 %d / 判未知 %d / 认成李云霄 %d   （共 %d 张）"
      % (sum(1 for r in rows if r[1] == B), sum(1 for r in rows if r[1] == "未知"),
         sum(1 for r in rows if r[1] == "李云霄"), len(rows)))

print()
print("=" * 74)
print("步骤 4 · 对照：鞠婧祎本人的照片此时还认得出来吗（用注入前读进内存的原图）")
print("=" * 74)
rows = []
for n, dt in orig_ju[:5]:
    rows.append((n,) + recog(dt, n))
print("    %-26s %-10s %8s %8s" % ("鞠婧祎的原图", "判成", "相似度", "margin"))
for fn, name, s, m in rows:
    print("    %-26s %-10s %8.4f %8.4f" % (fn[:26], name, s, m))
print("    → 认成鞠婧祎 %d / 判未知 %d   （共 %d 张）"
      % (sum(1 for r in rows if r[1] == B), sum(1 for r in rows if r[1] == "未知"), len(rows)))

print()
print("=" * 74)
print("实验结束，底库处于被污染状态，需要外部恢复")
print("=" * 74)
