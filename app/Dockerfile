FROM python:3.12-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    TZ=Asia/Shanghai \
    PIP_NO_CACHE_DIR=1

ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -i "$PIP_INDEX" -r requirements.txt

# 构建期闸门：OpenCV 人脸模块与 MySQL 驱动必须可用
RUN python -c "import cv2, numpy, fastapi, uvicorn, PIL, multipart, sqlalchemy, pymysql, cryptography; print('cv2 =', cv2.__version__); print('numpy =', numpy.__version__); assert hasattr(cv2, 'FaceDetectorYN'), 'missing FaceDetectorYN'; assert hasattr(cv2, 'FaceRecognizerSF'), 'missing FaceRecognizerSF'; print('OpenCV face module OK')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8", "--timeout-keep-alive", "65"]
