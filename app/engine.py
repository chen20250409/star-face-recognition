# -*- coding: utf-8 -*-
"""YuNet 人脸检测 + SFace 人脸识别 + Pillow 中文标注"""
import io
import os
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

log = logging.getLogger("engine")

DET_MODEL = os.getenv("DET_MODEL", "/models/face_detection_yunet_2023mar.onnx")
REC_MODEL = os.getenv("REC_MODEL", "/models/face_recognition_sface_2021dec.onnx")
FONT_PATH = os.getenv("FONT_PATH", "/app/fonts/simhei.ttf")

THRESHOLD = float(os.getenv("THRESHOLD", "0.363"))
MARGIN = float(os.getenv("MARGIN", "0.03"))
DET_SCORE = float(os.getenv("DET_SCORE", "0.55"))
MIN_SIDE = int(os.getenv("MIN_SIDE", "480"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))

UNKNOWN = "未知"


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class FaceEngine:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), DET_SCORE, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")
        cv2.setNumThreads(int(os.getenv("CV_THREADS", "2")))
        log.info("模型已加载 det=%s rec=%s", os.path.basename(DET_MODEL), os.path.basename(REC_MODEL))

    # ---------- 图像预处理 ----------
    @staticmethod
    def decode(data: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)          # 修正手机竖拍的 EXIF 旋转
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def prepare(self, img: Image.Image, extra_scale: float = 1.0):
        w, h = img.size
        s = 1.0
        short = min(w, h)
        if short < MIN_SIDE:                        # 小图放大，否则 YuNet 检不出
            s = MIN_SIDE / short
        if max(w, h) * s > MAX_SIDE:                # 大图缩小，省算力
            s *= MAX_SIDE / (max(w, h) * s)
        s *= extra_scale
        if abs(s - 1.0) > 0.01:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return bgr, img

    def _detect_once(self, bgr):
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        return faces

    def detect_all(self, data: bytes):
        """返回 (bgr, rgb_pil, faces)；小图会自动放大重试一次"""
        img = self.decode(data)
        bgr, rgb = self.prepare(img)
        faces = self._detect_once(bgr)
        if faces is None or len(faces) == 0:
            bgr2, rgb2 = self.prepare(img, extra_scale=1.5)
            faces2 = self._detect_once(bgr2)
            if faces2 is not None and len(faces2) > 0:
                log.info("首次未检出，放大 1.5 倍后检出 %d 张脸", len(faces2))
                bgr, rgb, faces = bgr2, rgb2, faces2
        if faces is None:
            faces = np.zeros((0, 15), dtype=np.float32)
        return bgr, rgb, faces

    # ---------- 特征 ----------
    def embed(self, bgr, face_row) -> np.ndarray:
        aligned = self.recognizer.alignCrop(bgr, face_row)
        feat = self.recognizer.feature(aligned)
        return _l2(feat.flatten().astype(np.float32))

    @staticmethod
    def largest_face(faces) -> np.ndarray:
        areas = faces[:, 2] * faces[:, 3]
        return faces[int(np.argmax(areas))]

    # ---------- 标注 ----------
    def annotate(self, rgb: Image.Image, results) -> Image.Image:
        img = rgb.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        fs = max(16, int(min(w, h) * 0.045))
        try:
            font = ImageFont.truetype(FONT_PATH, fs)
        except Exception as e:
            log.warning("字体加载失败 %s，回退默认字体（中文会是方块）", e)
            font = ImageFont.load_default()
        lw = max(2, int(min(w, h) * 0.005))
        for r in results:
            x, y, bw, bh = [int(round(v)) for v in r["bbox"]]
            known = not r["is_unknown"]
            color = (46, 170, 90) if known else (225, 70, 70)
            draw.rectangle([x, y, x + bw, y + bh], outline=color, width=lw)
            text = f"{r['name']} {r['score']:.2f}" if known else f"{r['name']} {r['score']:.2f}"
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            ty = y - th - 8 if y - th - 8 > 0 else y + bh + 6
            draw.rectangle([x, ty, x + tw + 10, ty + th + 8], fill=color)
            draw.text((x + 5, ty + 4), text, fill=(255, 255, 255), font=font)
        return img

    @staticmethod
    def to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
