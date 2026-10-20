# -*- coding: utf-8 -*-
"""物品检测自检：验证「人脸 + 物品」双检测链路

测试图放在宿主机 data/test/ 下（挂载进容器为 /data/test）：
  dog.jpg    经典 YOLO 测试图：狗 + 自行车 + 汽车
  horses.jpg 马 + 人 + 狗
"""
import io
import json
import os
import sys
import uuid
import urllib.error
import urllib.request

BASE = os.getenv("BASE", "http://127.0.0.1")
TEST_DIR = os.getenv("TEST_DIR", "/data/test")
GALLERY_DIR = os.getenv("GALLERY_DIR", "/data/gallery")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("  -> " + str(detail) if detail else ""), flush=True)
    return cond


def post(files, fields):
    b = "----v" + uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode())
    for field, fn, data in files:
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                    "Content-Type: image/jpeg\r\n\r\n" % (b, field, fn)).encode())
        body.write(data)
        body.write(b"\r\n")
    body.write(("--%s--\r\n" % b).encode())
    req = urllib.request.Request(BASE + "/api/recognize", data=body.getvalue(), method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + b)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def load(name):
    with open(os.path.join(TEST_DIR, name), "rb") as f:
        return f.read()


def show(tag, res):
    print("  %s 耗时 %dms" % (tag, res["elapsed_ms"]))
    print("    人脸:", [(x["name"], x["score"]) for x in res["faces"]])
    print("    物品:", [(x["label_cn"], x["label"], x["score"]) for x in res["objects"]])


print("=" * 70)
print("1) dog.jpg：应识别出 dog / bicycle / car 等")
print("=" * 70)
_, r = post([("files", "dog.jpg", load("dog.jpg"))], {"source": "verify", "detect_objects": "true"})
res = r["results"][0]
show("dog.jpg", res)
labels = [o["label"] for o in res["objects"]]
check("检出物品数 >= 2", len(res["objects"]) >= 2, len(res["objects"]))
check("识别出 dog", "dog" in labels, labels)
check("识别出交通工具", any(x in labels for x in ("bicycle", "car", "truck", "motorbike")), labels)
check("物品都带中文名", all(o["label_cn"] for o in res["objects"]), [o["label_cn"] for o in res["objects"]])
check("结果图路径正常", res["result"].startswith("/results/"), res["result"])

with urllib.request.urlopen(BASE + res["result"], timeout=30) as rr:
    blob = rr.read()
check("标注图可经 nginx 访问", rr.status == 200 and len(blob) > 5000, (rr.status, len(blob)))

print()
print("=" * 70)
print("2) horses.jpg：应识别出 horse 等")
print("=" * 70)
_, r2 = post([("files", "horses.jpg", load("horses.jpg"))], {"source": "verify", "detect_objects": "true"})
res2 = r2["results"][0]
show("horses.jpg", res2)
labels2 = [o["label"] for o in res2["objects"]]
check("识别出 horse", "horse" in labels2, labels2)

print()
print("=" * 70)
print("3) 关闭物品检测（detect_objects=false），objects 必须为空")
print("=" * 70)
_, r3 = post([("files", "dog.jpg", load("dog.jpg"))], {"source": "verify", "detect_objects": "false"})
res3 = r3["results"][0]
print("  耗时 %dms  物品数 %d" % (res3["elapsed_ms"], len(res3["objects"])))
check("objects 为空", len(res3["objects"]) == 0, res3["objects"])
check("关闭后不慢于开启", res3["elapsed_ms"] <= res["elapsed_ms"] + 30,
      (res3["elapsed_ms"], res["elapsed_ms"]))

print()
print("=" * 70)
print("4) 人像场景：人脸识别照常工作，物品检测不干扰")
print("=" * 70)
gal = os.path.join(GALLERY_DIR, "景甜")
fn = sorted(os.listdir(gal))[0]
with open(os.path.join(gal, fn), "rb") as f:
    pdata = f.read()
_, r4 = post([("files", fn, pdata)], {"source": "verify", "detect_objects": "true"})
res4 = r4["results"][0]
show("景甜照片", res4)
check("人脸仍识别为景甜", len(res4["faces"]) >= 1 and res4["faces"][0]["name"] == "景甜",
      res4["faces"][0]["name"] if res4["faces"] else None)
check("人脸框与人的物品框不重复堆叠",
      all(not (o["label"] == "person") for o in res4["objects"]),
      [o["label"] for o in res4["objects"]])

print()
print("=" * 70)
print("5) 记录接口是否带出物品")
print("=" * 70)
with urllib.request.urlopen(BASE + "/api/records?limit=4", timeout=30) as rr:
    recs = json.loads(rr.read().decode())["records"]
for x in recs[:4]:
    print("   #%s 人脸=%s 物品=%s" % (x["id"],
          [(f["name"], f["score"]) for f in x["faces"]],
          [(o["label_cn"], o["score"]) for o in x["objects"]]))
check("记录里有物品字段", any(len(x["objects"]) > 0 for x in recs), None)

print()
print("=" * 70)
print("结果：通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项:", FAIL)
print("=" * 70)
sys.exit(1 if FAIL else 0)
