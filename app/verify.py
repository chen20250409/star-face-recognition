# -*- coding: utf-8 -*-
"""端到端自检：全部请求经 nginx(80) 走一遍，验证 nginx→app→MySQL→静态目录 全链路"""
import io
import json
import os
import sys
import uuid
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

BASE = os.getenv("BASE", "http://127.0.0.1")
GALLERY_DIR = os.getenv("GALLERY_DIR", "/data/gallery")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("  -> " + str(detail) if detail else ""), flush=True)
    return cond


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def post_multipart(path, files, fields=None):
    b = "----verify" + uuid.uuid4().hex
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
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def del_person(pid):
    req = urllib.request.Request(BASE + "/api/persons/%d" % pid, method="DELETE")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def gallery_image(person, idx=0):
    d = os.path.join(GALLERY_DIR, person)
    names = sorted(n for n in os.listdir(d) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    with open(os.path.join(d, names[idx]), "rb") as f:
        return names[idx], f.read()


def noise_image():
    arr = (np.random.rand(400, 400, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


print("=" * 70)
print("1) 健康检查 /api/health（经 nginx）")
print("=" * 70)
st, body, _ = get("/api/health")
h = json.loads(body.decode())
print("  ", json.dumps(h, ensure_ascii=False))
check("HTTP 200", st == 200, st)
check("db_ok=true", h.get("db_ok") is True, h.get("db_ok"))
check("persons=3", h.get("persons") == 3, h.get("persons"))
check("samples>=24", h.get("samples", 0) >= 24, h.get("samples"))

print()
print("=" * 70)
print("2) 正样本识别：上传【景甜】底库图，应命中")
print("=" * 70)
fname, data = gallery_image("景甜")
st, r = post_multipart("/api/recognize", [("files", fname, data)], {"source": "verify"})
res = r["results"][0]
print("  ", json.dumps(res, ensure_ascii=False))
faces = res["faces"]
check("检出 1 张脸", len(faces) == 1, len(faces))
check("人名=景甜", faces and faces[0]["name"] == "景甜", faces[0]["name"] if faces else None)
check("相似度>=0.5", faces and faces[0]["score"] >= 0.5, faces[0]["score"] if faces else None)
check("is_unknown=false", faces and faces[0]["is_unknown"] is False, None)
positive_url = res["result"]

print()
print("=" * 70)
print("3) 未知判定：删除【鞠婧祎】后上传她的照片，应返回「未知」")
print("=" * 70)
persons = json.loads(get("/api/persons")[1].decode())["persons"]
ju = [p for p in persons if p["name"] == "鞠婧祎"][0]
# 删除人物会连底库目录一起清掉，先把图读进内存，后面还要用它重新注册
d = os.path.join(GALLERY_DIR, "鞠婧祎")
files = []
for n in sorted(os.listdir(d)):
    if n.lower().endswith((".jpg", ".jpeg", ".png")):
        with open(os.path.join(d, n), "rb") as f:
            files.append(("files", n, f.read()))
print("  预先读入:", len(files), "张")
r_del = del_person(ju["id"])
print("  删除:", r_del)
check("删除人物时底库目录一并清理", r_del.get("dir_removed") is True, r_del)
check("底库目录已不存在", not os.path.isdir(d), d)
fname2, data2 = files[0][1], files[0][2]
st, r2 = post_multipart("/api/recognize", [("files", fname2, data2)], {"source": "verify"})
res2 = r2["results"][0]
print("  ", json.dumps(res2, ensure_ascii=False))
f2 = res2["faces"]
check("人名=未知", f2 and f2[0]["name"] == "未知", f2[0]["name"] if f2 else None)
check("is_unknown=true", f2 and f2[0]["is_unknown"] is True, None)
check("不再返回 candidate（页面不暴露「最接近」）",
      f2 and "candidate" not in f2[0], sorted(f2[0].keys()) if f2 else None)

print()
print("=" * 70)
print("4) 网页注册链路：重新注册【鞠婧祎】（模拟网页上传，全量恢复）")
print("=" * 70)
print("  待注册张数:", len(files))
st, r3 = post_multipart("/api/persons", files, {"name": "鞠婧祎"})
print("  ", json.dumps(r3, ensure_ascii=False)[:300])
check("注册成功 added>=1", r3.get("added", 0) >= 1, r3.get("added"))
check("底库特征恢复到删前水平", r3.get("added", 0) >= len(files) - 1, (r3.get("added"), len(files)))
persons = json.loads(get("/api/persons")[1].decode())["persons"]
check("人物数回到 3", len(persons) == 3, [p["name"] for p in persons])

print()
print("=" * 70)
print("5) 无脸图：纯噪声图应检出 0 张脸")
print("=" * 70)
st, r4 = post_multipart("/api/recognize", [("files", "noise.jpg", noise_image())], {"source": "verify"})
check("faces 为空", len(r4["results"][0]["faces"]) == 0, len(r4["results"][0]["faces"]))

print()
print("=" * 70)
print("6) 静态目录与结果图（经 nginx 直读文件）")
print("=" * 70)
st, blob, ctype = get(positive_url)
check("结果图 GET 200", st == 200, st)
check("Content-Type=image/jpeg", "image/jpeg" in ctype, ctype)
check("结果图非空", len(blob) > 2000, len(blob))

idx = json.loads(get("/api/records?limit=1")[1].decode())["records"][0]
st, blob2, ctype2 = get(idx["original"])
check("原图可访问", st == 200 and len(blob2) > 1000, (st, len(blob2)))

st, page, ctype3 = get("/")
check("首页可访问", st == 200, st)
check("首页是中文页面", "人脸" in page.decode("utf-8", "ignore"), None)

print()
print("=" * 70)
print("7) 识别记录")
print("=" * 70)
recs = json.loads(get("/api/records?limit=10")[1].decode())["records"]
print("  记录条数:", len(recs))
for r in recs[:4]:
    print("   #%s %s %s %s %s" % (r["id"], r["created_at"], r["source"],
                                  [(f["name"], f["score"]) for f in r["faces"]], r["result"]))
check("记录数 >= 3", len(recs) >= 3, len(recs))

print()
print("=" * 70)
print("结果：通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项：", FAIL)
print("=" * 70)
sys.exit(1 if FAIL else 0)
