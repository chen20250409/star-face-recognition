# -*- coding: utf-8 -*-
"""人脸识别服务：FastAPI + YuNet/SFace + MySQL"""
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, func

from database import (FaceSample, Person, RecogFace, RecogLog, RecogObject,
                      SessionLocal, init_db, wait_for_db)
from engine import MARGIN, THRESHOLD, UNKNOWN, FaceEngine
from objects import ObjectDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="人脸识别系统", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA = os.getenv("DATA_DIR", "/data")
UPLOAD_DIR = os.path.join(DATA, "uploads")
RESULT_DIR = os.path.join(DATA, "results")
GALLERY_DIR = os.getenv("GALLERY_DIR", os.path.join(DATA, "gallery"))
MAX_FILES = int(os.getenv("MAX_FILES", "20"))
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

ENGINE: FaceEngine = None
OBJECTS: ObjectDetector = None


# ---------------------------------------------------------------- 底库缓存
class Gallery:
    """底库：按人取特征类中心（centroid），比对时用余弦相似度

    实验数据（留一法，27 样本 / 3 人）：
      max 聚合 72.2% | top3 均值 79.6% | centroid 83.3%  ← 选 centroid
      水平翻转 TTA 无增益（75.9%），不采用
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.centroids = np.zeros((0, 128), dtype=np.float32)
        self.pids = np.zeros((0,), dtype=np.int64)
        self.counts = {}
        self.names = {}

    def load(self):
        with SessionLocal() as s:
            self.names = {p.id: p.name for p in s.query(Person).all()}
            rows = s.query(FaceSample).all()
        if not rows:
            self.centroids = np.zeros((0, 128), dtype=np.float32)
            self.pids = np.zeros((0,), dtype=np.int64)
            self.counts = {}
            return
        groups = {}
        for r in rows:
            vec = np.frombuffer(r.embedding, dtype=np.float32)
            groups.setdefault(r.person_id, []).append(vec)
        pids, cents = [], []
        for pid, vs in groups.items():
            c = np.mean(np.stack(vs), axis=0)
            n = float(np.linalg.norm(c))
            pids.append(pid)
            cents.append(c / n if n > 0 else c)
        self.pids = np.array(pids, dtype=np.int64)
        self.centroids = np.stack(cents).astype(np.float32)
        self.counts = {pid: len(vs) for pid, vs in groups.items()}
        log.info("底库已载入：%d 人 / %d 个特征（centroid 聚合）", len(self.pids), len(rows))

    def match(self, v: np.ndarray):
        """返回 (person_id, 最高分, 次高分)"""
        if len(self.centroids) == 0:
            return None, 0.0, 0.0
        sims = self.centroids @ v
        order = np.argsort(sims)[::-1]
        best_pid = int(self.pids[order[0]])
        best = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else 0.0
        return best_pid, best, second


GALLERY = Gallery()


# ---------------------------------------------------------------- 工具
def get_or_create_person(name: str) -> int:
    with SessionLocal() as s:
        p = s.query(Person).filter(Person.name == name).first()
        if p is None:
            p = Person(name=name)
            s.add(p)
            s.commit()
            log.info("新建人物：%s (id=%d)", name, p.id)
        return p.id


def save_file(data: bytes, subdir: str, suffix: str) -> tuple:
    day = datetime.now().strftime("%Y-%m-%d")
    d = os.path.join(DATA, subdir, day) if subdir == "uploads" else os.path.join(DATA, subdir)
    os.makedirs(d, exist_ok=True)
    os.chmod(d, 0o755)
    name = uuid.uuid4().hex[:10] + suffix
    path = os.path.join(d, name)
    with open(path, "wb") as f:
        f.write(data)
    os.chmod(path, 0o644)
    url = f"/{subdir}/{day}/{name}" if subdir == "uploads" else f"/{subdir}/{name}"
    return path, url


def seed_gallery(eng: FaceEngine, clear: bool = False) -> dict:
    """扫描 /data/gallery/<人名>/ 建底库"""
    if clear:
        with SessionLocal() as s:
            s.query(FaceSample).delete()
            s.query(Person).delete()
            s.commit()
    added, failed = 0, []
    if not os.path.isdir(GALLERY_DIR):
        return {"added": 0, "failed": [], "reason": "目录不存在: %s" % GALLERY_DIR}
    for name in sorted(os.listdir(GALLERY_DIR)):
        pdir = os.path.join(GALLERY_DIR, name)
        if not os.path.isdir(pdir):
            continue
        pid = get_or_create_person(name)
        for fn in sorted(os.listdir(pdir)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            path = os.path.join(pdir, fn)
            try:
                with open(path, "rb") as f:
                    data = f.read()
                bgr, _rgb, faces = eng.detect_all(data)
                if len(faces) == 0:
                    failed.append({"file": f"{name}/{fn}", "reason": "未检测到人脸"})
                    continue
                row = eng.largest_face(faces)
                vec = eng.embed(bgr, row)
                with SessionLocal() as s:
                    s.add(FaceSample(person_id=pid, image_path=path,
                                     embedding=vec.tobytes(), det_score=float(row[14])))
                    s.commit()
                added += 1
            except Exception as e:
                failed.append({"file": f"{name}/{fn}", "reason": str(e)[:120]})
    return {"added": added, "failed": failed}


def recognize_one(data: bytes, source: str, detect_objects: bool = True) -> dict:
    t0 = time.time()
    bgr, rgb, faces = ENGINE.detect_all(data)
    results = []
    face_boxes = []
    for i in range(len(faces)):
        row = faces[i]
        vec = ENGINE.embed(bgr, row)
        pid, score, second = GALLERY.match(vec)
        candidate = GALLERY.names.get(pid) if pid else None
        margin = score - second
        # 双条件：分数过阈值 且 与第二名有足够间隔，否则判未知
        if pid and score >= THRESHOLD and margin >= MARGIN:
            name, unknown = candidate, False
        else:
            name, unknown = UNKNOWN, True
        bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        face_boxes.append(bbox)
        results.append({
            "bbox": bbox,
            "name": name,
            "score": round(float(score), 4),
            "margin": round(float(margin), 4),
            "is_unknown": unknown,
            "candidate": candidate,
            "det_score": round(float(row[14]), 4),
        })

    objects = []
    if detect_objects and OBJECTS is not None:
        objects = OBJECTS.detect(bgr, face_boxes)

    _op, orig_url = save_file(data, "uploads", ".jpg")
    annotated = ENGINE.annotate(rgb, results, objects)
    _rp, res_url = save_file(ENGINE.to_jpeg_bytes(annotated), "results", ".jpg")
    elapsed = int((time.time() - t0) * 1000)

    with SessionLocal() as s:
        row = RecogLog(source=source, original_path=orig_url, result_path=res_url,
                       elapsed_ms=elapsed, face_count=len(results))
        s.add(row)
        s.flush()
        log_id = row.id
        for r in results:
            s.add(RecogFace(log_id=log_id,
                            person_name=None if r["is_unknown"] else r["name"],
                            score=r["score"],
                            is_unknown=1 if r["is_unknown"] else 0,
                            candidate_name=r["candidate"],
                            bbox="%d,%d,%d,%d" % tuple(int(round(x)) for x in r["bbox"]),
                            det_score=r["det_score"]))
        for o in objects:
            s.add(RecogObject(log_id=log_id, label=o["label"], label_cn=o["label_cn"],
                              score=o["score"],
                              bbox="%d,%d,%d,%d" % tuple(int(round(x)) for x in o["bbox"])))
        s.commit()
    return {"log_id": log_id, "original": orig_url, "result": res_url,
            "elapsed_ms": elapsed, "faces": results, "objects": objects}


# ---------------------------------------------------------------- 启动
@app.on_event("startup")
def on_startup():
    global ENGINE, OBJECTS
    os.umask(0o022)
    for d in (UPLOAD_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o755)
    if not wait_for_db():
        raise RuntimeError("无法连接 MySQL，请检查数据库配置")
    init_db()
    ENGINE = FaceEngine()
    try:
        OBJECTS = ObjectDetector()
    except Exception as e:
        log.warning("物品检测模型加载失败（人脸功能不受影响）：%s", e)
        OBJECTS = None
    GALLERY.load()
    if len(GALLERY.names) == 0 and os.path.isdir(GALLERY_DIR):
        r = seed_gallery(ENGINE)
        GALLERY.load()
        log.info("自动建库完成：成功 %d 张，失败 %d 张 %s", r["added"], len(r["failed"]), r["failed"][:3])
    log.info("服务就绪，阈值 THRESHOLD=%.3f", THRESHOLD)


# ---------------------------------------------------------------- 接口
@app.get("/api/health")
def health():
    try:
        with SessionLocal() as s:
            persons = s.query(func.count(Person.id)).scalar()
            samples = s.query(func.count(FaceSample.id)).scalar()
        db_ok = True
    except Exception as e:
        log.warning("健康检查读库失败: %s", e)
        persons, samples, db_ok = 0, 0, False
    return {"ok": True, "db_ok": db_ok, "persons": persons, "samples": samples,
            "threshold": THRESHOLD, "margin": MARGIN,
            "engine_ready": ENGINE is not None, "objects_ready": OBJECTS is not None}


@app.post("/api/recognize")
async def recognize(files: List[UploadFile] = File(...), source: str = Form("upload"),
                    detect_objects: bool = Form(True)):
    if not files:
        raise HTTPException(400, "没有上传文件")
    if len(files) > MAX_FILES:
        raise HTTPException(400, "一次最多 %d 张" % MAX_FILES)
    out, errors = [], []
    for f in files:
        try:
            data = await f.read()
            ENGINE.decode(data)          # 先验证确实是图片
            out.append({**recognize_one(data, source, detect_objects),
                        "filename": f.filename})
        except Exception as e:
            log.warning("处理 %s 失败: %s", f.filename, e)
            errors.append({"filename": f.filename, "reason": str(e)[:160]})
    return {"ok": True, "results": out, "errors": errors}


@app.get("/api/persons")
def list_persons():
    with SessionLocal() as s:
        rows = (s.query(Person.id, Person.name, func.count(FaceSample.id))
                .outerjoin(FaceSample, FaceSample.person_id == Person.id)
                .group_by(Person.id, Person.name).all())
    return {"ok": True, "persons": [{"id": i, "name": n, "samples": c} for i, n, c in rows]}


@app.post("/api/persons")
async def add_person(name: str = Form(...), files: List[UploadFile] = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "姓名不能为空")
    if len(name) > 64:
        raise HTTPException(400, "姓名过长")
    pid = get_or_create_person(name)
    added, failed = 0, []
    for f in files:
        try:
            data = await f.read()
            ENGINE.decode(data)
            bgr, _rgb, faces = ENGINE.detect_all(data)
            if len(faces) == 0:
                failed.append({"file": f.filename, "reason": "未检测到人脸"})
                continue
            row = ENGINE.largest_face(faces)
            vec = ENGINE.embed(bgr, row)
            with SessionLocal() as s:
                s.add(FaceSample(person_id=pid, image_path="web-upload:%s" % f.filename,
                                 embedding=vec.tobytes(), det_score=float(row[14])))
                s.commit()
            added += 1
        except Exception as e:
            failed.append({"file": f.filename, "reason": str(e)[:160]})
    GALLERY.load()
    return {"ok": True, "person_id": pid, "name": name, "added": added, "failed": failed}


@app.delete("/api/persons/{pid}")
def delete_person(pid: int):
    with SessionLocal() as s:
        p = s.query(Person).filter(Person.id == pid).first()
        if p is None:
            raise HTTPException(404, "人物不存在")
        name = p.name
        s.query(FaceSample).filter(FaceSample.person_id == pid).delete()
        s.delete(p)
        s.commit()
    GALLERY.load()
    return {"ok": True, "deleted": name}


@app.post("/api/gallery/rebuild")
def rebuild():
    r = seed_gallery(ENGINE, clear=True)
    GALLERY.load()
    return {"ok": True, **r}


@app.get("/api/records")
def records(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    with SessionLocal() as s:
        logs = s.query(RecogLog).order_by(desc(RecogLog.id)).limit(limit).all()
        out = []
        for L in logs:
            faces = s.query(RecogFace).filter(RecogFace.log_id == L.id).all()
            objs = s.query(RecogObject).filter(RecogObject.log_id == L.id).all()
            out.append({
                "id": L.id,
                "created_at": L.created_at.strftime("%Y-%m-%d %H:%M:%S") if L.created_at else "",
                "source": L.source,
                "original": L.original_path,
                "result": L.result_path,
                "elapsed_ms": L.elapsed_ms,
                "face_count": L.face_count,
                "faces": [{"name": f.person_name or UNKNOWN, "score": f.score,
                           "is_unknown": bool(f.is_unknown), "candidate": f.candidate_name}
                          for f in faces],
                "objects": [{"label": o.label, "label_cn": o.label_cn, "score": o.score}
                            for o in objs],
            })
    return {"ok": True, "records": out}
