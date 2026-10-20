# -*- coding: utf-8 -*-
"""底库照片管理自检：清单 / 追加 / 替换 / 删除 + /gallery/ 静态访问

全部请求经 nginx(80) 走一遍，文件是否真的落盘在容器内直接查磁盘。
"""
import io
import json
import os
import sys
import uuid
import urllib.error
import urllib.request

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


def api(path):
    return json.loads(get(path)[1].decode())


def post_multipart(path, files, fields=None):
    b = "----vp" + uuid.uuid4().hex
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


def delete(path):
    req = urllib.request.Request(BASE + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def pids():
    return {p["name"]: p["id"] for p in api("/api/persons")["persons"]}


def photos(pid):
    return api("/api/persons/%d/photos" % pid)


def read_img(person, idx):
    d = os.path.join(GALLERY_DIR, person)
    names = sorted(n for n in os.listdir(d) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    n = names[idx % len(names)]
    with open(os.path.join(d, n), "rb") as f:
        return n, f.read()


print("=" * 70)
print("1) 照片清单：每个人每张图都要能看、能对到磁盘文件")
print("=" * 70)
db = pids()
print("  人物：", db)
pid = db["景甜"]
j = photos(pid)
print("  person:", j["person"], " 张数:", len(j["photos"]), " 缺失:", j["missing"])
check("photos 接口 ok", j.get("ok") is True, None)
check("照片数 == 特征数", len(j["photos"]) == 9, len(j["photos"]))
check("每张都有 url", all(p["url"] for p in j["photos"]), None)
check("每张文件都在磁盘上", all(p["exists"] for p in j["photos"]), j["missing"])
check("无缺失原图（历史 web-upload 路径已修复）", j["missing"] == 0, j["missing"])
print("  第一张：", j["photos"][0]["file"], "->", j["photos"][0]["url"])

st, blob, ctype = get(j["photos"][0]["url"])
check("/gallery/ 静态可访问", st == 200 and len(blob) > 1000, (st, len(blob)))
check("Content-Type 是图片", ctype.startswith("image/"), ctype)

print()
print("=" * 70)
print("2) 追加一张照片")
print("=" * 70)
before = len(j["photos"])
fn, data = read_img("景甜", 1)
st, r = post_multipart("/api/persons/%d/photos" % pid, [("files", fn, data)])
print("  ", r)
check("追加成功 added=1", r.get("added") == 1, r.get("added"))
j2 = photos(pid)
check("照片数 +1", len(j2["photos"]) == before + 1, (before, len(j2["photos"])))
new = j2["photos"][-1]
print("  新照片：", new["file"], "sid=", new["sid"])
check("新照片文件存在", new["exists"] and os.path.isfile(os.path.join(GALLERY_DIR, new["file"])), new["file"])
check("新照片可访问", get(new["url"])[0] == 200, new["url"])

print()
print("=" * 70)
print("3) 替换这张（应换图、换 sid、旧文件被清理）")
print("=" * 70)
fn2, data2 = read_img("景甜", 3)
st, r2 = post_multipart("/api/persons/%d/photos" % pid, [("files", fn2, data2)],
                        {"replace_id": new["sid"]})
print("  ", r2)
check("替换成功 replaced=true", r2.get("replaced") is True, r2.get("replaced"))
check("替换只算 1 张", r2.get("added") == 1, r2.get("added"))
j3 = photos(pid)
check("照片总数不变", len(j3["photos"]) == before + 1, len(j3["photos"]))
check("旧 sid 已消失", all(p["sid"] != new["sid"] for p in j3["photos"]),
      [p["sid"] for p in j3["photos"]])
check("旧文件已删除", not os.path.isfile(os.path.join(GALLERY_DIR, new["file"])), new["file"])
cur = j3["photos"][-1]
check("替换后的图能访问", get(cur["url"])[0] == 200, cur["url"])

print()
print("=" * 70)
print("4) 删除这张")
print("=" * 70)
st, r3 = delete("/api/persons/%d/photos/%d" % (pid, cur["sid"]))
print("  ", r3)
check("删除成功", r3.get("ok") is True, None)
check("剩余数量正确", r3.get("left") == before, (r3.get("left"), before))
check("文件被清理", r3.get("file_removed") is True, r3.get("file_removed"))
j4 = photos(pid)
check("照片数回到原始", len(j4["photos"]) == before, len(j4["photos"]))

print()
print("=" * 70)
print("5) 新注册的人：图必须真落盘，删到 0 张人物仍在")
print("=" * 70)
tmpname = "自检临时人"
st, r4 = post_multipart("/api/persons", [("files", "t.jpg", data)], {"name": tmpname})
print("  ", r4)
check("临时人物注册成功", r4.get("added") == 1, r4.get("added"))
tpid = r4["person_id"]
tp = photos(tpid)
check("有 1 张照片", len(tp["photos"]) == 1, len(tp["photos"]))
check("照片有原图（注册已落盘）", tp["photos"][0]["exists"], tp["photos"][0])
abs_tmp = os.path.join(GALLERY_DIR, tp["photos"][0]["file"])
check("文件确实在 gallery 目录下", os.path.isfile(abs_tmp), abs_tmp)
st, r5 = delete("/api/persons/%d/photos/%d" % (tpid, tp["photos"][0]["sid"]))
check("删到 0 张", r5.get("left") == 0, r5.get("left"))
check("人物仍存在", tmpname in pids(), list(pids().keys()))

st, r6 = delete("/api/persons/%d" % tpid)
print("  删除临时人物:", r6)
check("清理临时人物", r6.get("ok") is True, None)
check("临时目录一并删除", not os.path.isdir(os.path.join(GALLERY_DIR, tmpname)), tmpname)
check("人物数回到 3", len(pids()) == 3, list(pids().keys()))

print()
print("=" * 70)
print("6) 底库缓存同步（改完照片立即可用，无需重启）")
print("=" * 70)
h = api("/api/health")
print("  ", json.dumps(h, ensure_ascii=False))
check("特征总数回到 27", h["samples"] == 27, h["samples"])
check("人物数 3", h["persons"] == 3, h["persons"])

print()
print("=" * 70)
print("结果：通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项：", FAIL)
print("=" * 70)
sys.exit(1 if FAIL else 0)
