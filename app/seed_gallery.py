# -*- coding: utf-8 -*-
"""手动重建底库：docker compose exec app python seed_gallery.py"""
import json
import logging

from database import init_db, wait_for_db
from engine import FaceEngine
from main import GALLERY, GALLERY_DIR, seed_gallery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if not wait_for_db():
    raise SystemExit("连不上 MySQL")
init_db()
eng = FaceEngine()
res = seed_gallery(eng, clear=True)
GALLERY.load()
print(json.dumps({"gallery_dir": GALLERY_DIR, **res}, ensure_ascii=False, indent=2))
