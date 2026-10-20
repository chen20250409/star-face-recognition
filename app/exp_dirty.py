# -*- coding: utf-8 -*-
"""脏数据实验：把【景甜】的照片以【鞠婧祎】的名义注册进底库，看景甜会被判成谁

流程（全部走真实接口，不碰内部变量）：
  阶段 0  干净底库，景甜留出图逐张识别 —— 基线
  阶段 1  往「鞠婧祎」名下注入 4 张景甜照片（轻度污染）
  阶段 2  删掉「鞠婧祎」原本的 9 张，只剩景甜那 4 张（重度污染）
  阶段 3  对照：鞠婧祎本人的照片此时会被判成谁

不负责恢复，恢复由外部 mysqldump + tar 完成。
"""
import io
import json
import os
import sys
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


def table(rows):
    print("    %-26s %-10s %8s %8s" % ("查询图", "判成", "相似度", "margin"))
    for fn, name, s, m in rows:
        print("    %-26s %-10s %8.4f %8.4f" % (fn[:26], name, s, m))
    hit = sum(1 for r in rows if r[1] == A)
    unk = sum(1 for r in rows if r[1] == "未知")
    bad = sum(1 for r in rows if r[1] == B)
    print("    → 认成景甜 %d / 判未知 %d / 认成鞠婧祎 %d   （共 %d 张）"
          % (hit, unk, bad, len(rows)))


d = pids()
print("=" * 74)
print("初始底库")
print("=" * 74)
for k, v in d.items():
    print("  %-6s id=%-3d 特征 %d 张" % (k, v, len(photos(v))))

jing_dir = os.path.join(GALLERY, A)
names = sorted(n for n in os.listdir(jing_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))
inject = names[:4]
query = names[4:]
print("\n  景甜共 %d 张：注入用 %d 张，查询用 %d 张（两者不重叠，避免自己跟自己比）"
      % (len(names), len(inject), len(query)))
q = [(n, open(os.path.join(jing_dir, n), "rb").read()) for n in query]

print()
print("=" * 74)
print("阶段 0 · 干净底库基线：景甜的查询图应稳定判成「景甜」")
print("=" * 74)
table([(n,) + recog(dt, n) for n, dt in q])

b_pid = d[B]
orig_b = [p["sid"] for p in photos(b_pid)]
print()
print("=" * 74)
print("阶段 1 · 轻度污染：把景甜的 %d 张以「鞠婧祎」名义注册进去" % len(inject))
print("=" * 74)
for n in inject:
    dt = open(os.path.join(jing_dir, n), "rb").read()
    r = post_multipart("/api/persons/%d/photos" % b_pid, [("files", n, dt)])
    print("    注入 %-30s added=%s" % (n[:30], r.get("added")))
print("    鞠婧祎现在特征数：%d（原 %d）" % (len(photos(b_pid)), len(orig_b)))
table([(n,) + recog(dt, n) for n, dt in q])

print()
print("=" * 74)
print("阶段 2 · 重度污染：删掉「鞠婧祎」原本的 %d 张，只剩景甜那 %d 张" % (len(orig_b), len(inject)))
print("=" * 74)
for sid in orig_b:
    delete("/api/persons/%d/photos/%d" % (b_pid, sid))
print("    鞠婧祎现在特征数：%d" % len(photos(b_pid)))
table([(n,) + recog(dt, n) for n, dt in q])

print()
print("=" * 74)
print("阶段 3 · 对照：鞠婧祎本人的照片，此时会被判成谁")
print("=" * 74)
ju_dir = os.path.join(GALLERY, B)
ju_names = sorted(n for n in os.listdir(ju_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))[:5]
rows = []
for n in ju_names:
    dt = open(os.path.join(ju_dir, n), "rb").read()
    rows.append((n,) + recog(dt, n))
print("    %-26s %-10s %8s %8s" % ("鞠婧祎的照片", "判成", "相似度", "margin"))
for fn, name, s, m in rows:
    print("    %-26s %-10s %8.4f %8.4f" % (fn[:26], name, s, m))
print("    → 判成鞠婧祎 %d / 判未知 %d / 判成景甜 %d"
      % (sum(1 for r in rows if r[1] == B), sum(1 for r in rows if r[1] == "未知"),
         sum(1 for r in rows if r[1] == A)))

print()
print("=" * 74)
print("实验结束，底库处于被污染状态，需要外部恢复")
print("=" * 74)
