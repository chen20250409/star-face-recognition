# 人脸识别 + 物品检测系统（YuNet + SFace + YOLOv4-tiny + MySQL + nginx）

上传图片 / 摄像头抓拍 → 后端比对人脸底库 → 返回「人名 + 相似度」，同时用 YOLOv4-tiny 检出画面里的物品（猫、狗、汽车……COCO 80 类），原图与标注结果图落静态目录，识别记录写入 MySQL。

## 零、从仓库克隆后怎么跑起来

仓库里**不包含**大体积/有版权风险的资源，需要补齐三步：

```bash
git clone <repo> face-app && cd face-app

# 1) 配置凭据
cp .env.example .env && vi .env          # 填 DB_PASS

# 2) 下载模型（约 63MB：人脸 2 个 + 物品 3 个，带 sha256 校验）
python3 tools/dl_models.py

# 3) 放中文字体（微软版权字体，未随仓库分发）
#    任选一个中文字体放到 app/fonts/simhei.ttf
#    缺字体也能跑，只是结果图上的中文会显示成方块

# 4) 放底库照片（每人一个子目录，目录名即人名）
#    data/gallery/张三/1.jpg ...

docker compose up -d
```

未入库的资源及原因：

| 资源 | 原因 | 补齐方式 |
|---|---|---|
| `models/*.onnx` | 人脸模型 39MB 二进制 | `tools/dl_models.py` 自动下载并校验 |
| `models/yolov4-tiny.*` | 物品模型 24MB 权重 + cfg + 类名 | 同上 |
| `app/fonts/*.ttf` | 微软黑体版权，不可再分发 | 自行放置 |
| `data/gallery/*` | 底库人像涉及肖像权 | 自行准备 |
| `.env` | 含数据库密码 | 从 `.env.example` 复制 |

## 一、组成

| 组件 | 说明 |
|---|---|
| app | Python 3.12 + FastAPI + OpenCV（YuNet 人脸检测 / SFace 识别 / YOLOv4-tiny 物品检测），监听 8000，**host 网络** |
| nginx | 反向代理 `/api/` → app，静态直服 `/uploads/`、`/results/`、`/gallery/`、前端页面，监听 80 |
| MySQL | 复用宿主机 MySQL 8.4.11，库 `faces`，用户 `face`，utf8mb4 |

底库：`face_samples.embedding` 存 128 维 float32（512 字节）；识别时按人求**类中心（centroid）**后做余弦比对。

物品检测：`recognition_objects` 表按 `log_id` 存每张图的物品列表（英文类名、中文名、置信度、bbox）。

## 二、访问方式

浏览器无法直连虚拟机 80 端口（VMware 只转发了 22→2222），需要 SSH 隧道：

```bash
ssh -N -L 8080:127.0.0.1:80 -p 2222 chen@127.0.0.1
# 保持窗口不关，浏览器打开 http://localhost:8080
```

Windows 上直接双击 `face-tunnel.cmd` 即可。

## 三、目录

```
face-app/
├── docker-compose.yml
├── models/                 # YuNet / SFace onnx（只读挂载，已校验 sha256）
├── app/
│   ├── main.py             # 路由 + 底库缓存 + 识别流程
│   ├── engine.py           # 人脸检测 / 特征 / 中文标注（人脸框 + 物品框）
│   ├── objects.py          # 物品检测（YOLOv4-tiny，COCO 80 类）
│   ├── database.py         # SQLAlchemy 表定义
│   ├── seed_gallery.py     # 扫描 data/gallery 重建底库
│   ├── calibrate.py        # 留一法看分数分布、建议阈值
│   ├── experiment.py       # 聚合方式 × TTA 对比实验
│   ├── experiment2.py      # 阈值 + margin 决策规则搜索
│   ├── verify.py           # 端到端自检（24 项断言）
│   ├── verify_objects.py   # 物品检测自检（12 项断言）
│   ├── verify_photos.py    # 人脸库照片管理自检（32 项断言）
│   └── fonts/simhei.ttf
├── web/index.html          # 单页前端（4 个 tab）
├── nginx/default.conf
└── data/
    ├── gallery/<人名>/*.jpg   # 底库原图（27 张 / 3 人）
    ├── uploads/YYYY-MM-DD/    # 上传原图
    ├── results/               # 标注结果图
    └── test/                  # 自检用样例图（dog.jpg / horses.jpg）
```

## 四、常用命令

```bash
cd ~/face-app

sudo docker compose up -d                  # 启停
sudo docker compose ps
sudo docker compose logs -f --tail=100 app

sudo docker compose build app && sudo docker compose up -d    # 改代码后重建

sudo docker compose exec -T app python seed_gallery.py        # 重建底库
sudo docker compose exec -T app python calibrate.py           # 看分数分布
sudo docker compose exec -T app python verify.py              # 人脸端到端自检
sudo docker compose exec -T app python verify_objects.py      # 物品检测自检
sudo docker compose exec -T app python verify_photos.py       # 人脸库照片管理自检
sudo docker stats --no-stream                                 # 资源占用
```

## 五、识别判定规则

一个检测框被判为「已知」需要**同时**满足：

1. 与某人类中心的余弦相似度 ≥ `THRESHOLD`（默认 0.363）
2. 与第二名的差距 ≥ `MARGIN`（默认 0.03）

否则判为「未知」，同时把最接近的候选记进 `candidate_name` 便于排查。

### 聚合方式选择的实验依据

留一法（27 样本 / 3 人，每次剔除自身后比对）：

| 方案 | 最佳准确率 |
|---|---|
| 每人取样本最大值 max | 72.2% |
| top3 均值 | 79.6% |
| **类中心 centroid（采用）** | **83.3%** |
| centroid + 水平翻转 TTA | 75.9%（无增益，未采用） |

阈值没有压到"让这份样本最好看"的位置（那样对真陌生人会失效），保持在官方 0.363 并叠加 margin。

底库质量说明：这 27 张是 360px 级微博缩略图，带水印、姿态差异大（尤其鞠婧祎），留一法仍有 2/27 的真实误判，属模型能力上限。换成清晰正脸照能明显提升。

## 六、物品检测

除了认人，还会回答"画面里还有什么东西"，覆盖 COCO 80 类（人、猫、狗、鸟、汽车、自行车、卡车、瓶子、手机、书、椅子、盆栽、电视、笔记本电脑……）。

### 选型：YOLOv4-tiny + `cv2.dnn`，不用 torch

| 方案 | 取舍 |
|---|---|
| **YOLOv4-tiny（采用）** | 权重 23MB，只依赖已有的 OpenCV，CPU 单帧 120~180ms，镜像不用变大 |
| YOLOv8 + ultralytics | 精度更高，但要装 torch（+800MB），无 GPU 时反而更慢 |
| MobileNet-SSD | 更快但小目标漏检明显 |

模型在 `models/` 下，由 `tools/dl_models.py` 下载（权重走 `gh-proxy.com` 代理，因为 `objects.githubusercontent.com` 在部分网络下被拦）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `yolov4-tiny.cfg` | 170KB | 网络结构 |
| `yolov4-tiny.weights` | 23.1MB | 权重 |
| `coco.names` | 620B | 80 个英文类名，顺序与 `objects.py` 里的 `CN_NAMES` 严格对应 |

### 人的框与人脸框的去重（实测数据）

YOLO 的 `person` 框是全身/半身框，人脸框只是其中一小块，两者 **IoU 只有 0.04~0.09**，用 IoU 判断"这是同一个人"永远不成立，结果就是人脸框外套一个巨大的蓝框。

实测 6 张底库人像：

| 图片 | 人脸框面积 | person 框面积 | IoU | 人脸被 person 框覆盖 | 面积比 |
|---|---|---|---|---|---|
| 景甜 01 | 6240 | 137159 | 0.045 | 100% | 22.0× |
| 景甜 02 | 10366 | 267462 | 0.039 | 100% | 25.8× |
| 李云霄 01 | 37253 | 414312 | 0.090 | 100% | 11.1× |

所以判定条件改成**包含比例**：人脸框有 ≥ `OBJ_FACE_COVER`（默认 0.85）的面积落在 person 框内，就认为这个 person 框对应的正是已识别的这张脸，不再重复标注。没检出人脸的人（背景里的路人）仍然保留 person 框。

### 实测效果

| 图片 | 检出结果 | 耗时 |
|---|---|---|
| `dog.jpg` | 狗 0.81 / 卡车 0.75 / 自行车 0.37 | 373ms |
| `horses.jpg` | 牛 0.95 / 马 0.79 / 马 0.52 | 184ms |
| 人像（景甜） | 人脸「景甜 0.71」，无冗余 person 框 | 162ms |

关闭物品检测（网页勾选框取消 / API 传 `detect_objects=false`）后单张约 110ms，不想要这部分开销时可以直接关。

## 七、可调参数（docker-compose.yml → app.environment）

| 变量 | 默认 | 说明 |
|---|---|---|
| `THRESHOLD` | 0.363 | 余弦相似度阈值 |
| `MARGIN` | 0.03 | 与第二名的最大允许差距 |
| `DET_SCORE` | 0.55 | 人脸检测置信度阈值，漏检多则降到 0.45 |
| `MIN_SIDE` | 480 | 短边低于此值放大后再检测 |
| `MAX_SIDE` | 1280 | 长边超过此值缩小，省 CPU |
| `MAX_FILES` | 20 | 单次请求最多图片数 |
| `OBJ_SCORE` | 0.25 | 物品置信度阈值，漏检多就降到 0.15，误检多就升到 0.35 |
| `OBJ_NMS` | 0.40 | 物品框 NMS 去重阈值 |
| `OBJ_SIZE` | 416 | 网络输入边长，改 320 更快、608 更准也更慢 |
| `OBJ_MAX` | 12 | 单图最多返回几个物品 |
| `OBJ_FACE_COVER` | 0.85 | 人脸框被 person 框覆盖多少比例就丢弃该 person 框 |
| `CV_THREADS` | 2 | OpenCV 线程数，虚拟机核少时调低避免抢 CPU |

## 八、数据库

首次部署需先建库建用户（密码与 `.env` 里的 `DB_PASS` 保持一致）：

```sql
CREATE DATABASE IF NOT EXISTS faces DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'face'@'%' IDENTIFIED BY '<你的密码>';
GRANT ALL PRIVILEGES ON faces.* TO 'face'@'%';
FLUSH PRIVILEGES;
```

表结构由应用启动时自动创建（`database.py` 里的 SQLAlchemy 模型）。

```bash
sudo mysql -e "SELECT id,name FROM faces.persons;"
sudo mysql -e "SELECT id,created_at,source,face_count,elapsed_ms FROM faces.recognition_logs ORDER BY id DESC LIMIT 10;"
sudo mysql -e "SELECT log_id,person_name,score,is_unknown,candidate_name FROM faces.recognition_faces ORDER BY id DESC LIMIT 10;"
sudo mysql -e "SELECT log_id,label_cn,label,score FROM faces.recognition_objects ORDER BY id DESC LIMIT 10;"

sudo mysqldump --single-transaction faces > ~/faces-$(date +%F).sql   # 备份
```

Navicat：`ssh -L 3307:127.0.0.1:3306 chen@127.0.0.1 -p 2222`，再连 `127.0.0.1:3307`。

## 九、摄像头说明

摄像头代码完整保留，但浏览器要求**安全上下文**才允许 `getUserMedia`：

- `http://localhost:8080`（SSH 隧道）→ 可用
- `http://<IP>` → 浏览器禁用，页面给出明确提示
- 设备无摄像头 → 提示「未检测到摄像头设备」，改用「上传图片识别」

## 十、人脸库照片管理

网页「人脸库管理」里每行有一个「照片」按钮，点开能看到这个人底库里的每一张图，可以**逐张替换或删除**，也能继续追加。特征跟着一起重算，不用重建镜像、不用重启。

### 接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/persons/{pid}/photos` | 列出某人的照片：`sid`、`file`、`url`、`exists`、`det_score` |
| POST | `/api/persons/{pid}/photos` | 追加照片（`files`）；带 `replace_id` 时替换那一张（只能 1 张） |
| DELETE | `/api/persons/{pid}/photos/{sid}` | 删除单张，库里特征与磁盘文件一起清 |

照片通过 nginx 的 `/gallery/` 直接读静态文件（`data/gallery` 只读挂载进 nginx 容器）。

### 几条行为约定

- **注册图现在会真正落盘**：`/api/persons` 把图写到 `data/gallery/<人名>/<uuid>.jpg`，`image_path` 存绝对路径。早期版本只记了 `web-upload:文件名` 而没存文件，启动时会按文件名去 `gallery/<人名>/` 找回并改写路径（幂等，日志里有「已修复 N 条历史样本的图片路径」）。
- **替换 = 先加新的、再删旧的**：新图入库成功后才删旧记录，替换失败不会把已有照片弄丢；旧文件在无其它样本引用时一并清理。
- **删单张**：允许删到 0 张，此时人物仍在库里但匹配不到，前端会提示。
- **删整人**：连同 `gallery/<人名>/` 目录一起删，否则 `/api/gallery/rebuild` 会把人扫回来。原始训练图请另存（本地 `桌面\训练照片` 有备份）。
- 每次增删改后 `Gallery.load()` 重算类中心，识别立刻按新底库走。

自检：`verify_photos.py` 覆盖清单 / 追加 / 替换 / 删除 / 注册落盘 / 缓存同步，共 32 项。

## 十一、排障

| 现象 | 处理 |
|---|---|
| 页面 502 | `docker compose ps` 看 app 是否 healthy；`docker compose logs app` 查连库 |
| 连不上 MySQL | `mysql -uface -p -h127.0.0.1 faces` 手动试密码；app 必须用 host 网络才能访问 127.0.0.1:3306 |
| 中文显示方块 | 确认 `app/fonts/simhei.ttf` 在镜像内 |
| 某人总识别不出 | 底库图太少/太糊，在网页「人脸库管理」补 3 张以上清晰正脸照 |
| 结果图 404 | 确认 `data/uploads`、`data/results` 存在且 0755 |
