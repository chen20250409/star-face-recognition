# 人脸识别系统（YuNet + SFace + MySQL + nginx）

上传图片 / 摄像头抓拍 → 后端比对人脸底库 → 返回「人名 + 相似度」，原图与标注结果图落静态目录，识别记录写入 MySQL。

## 零、从仓库克隆后怎么跑起来

仓库里**不包含**大体积/有版权风险的资源，需要补齐三步：

```bash
git clone <repo> face-app && cd face-app

# 1) 配置凭据
cp .env.example .env && vi .env          # 填 DB_PASS

# 2) 下载模型（约 39MB，走 raw.githubusercontent 的 LFS 直链并校验 sha256）
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
| `models/*.onnx` | 39MB 二进制 | `tools/dl_models.py` 自动下载并校验 |
| `app/fonts/*.ttf` | 微软黑体版权，不可再分发 | 自行放置 |
| `data/gallery/*` | 底库人像涉及肖像权 | 自行准备 |
| `.env` | 含数据库密码 | 从 `.env.example` 复制 |

## 一、组成

| 组件 | 说明 |
|---|---|
| app | Python 3.12 + FastAPI + OpenCV（YuNet 检测 / SFace 识别），监听 8000，**host 网络** |
| nginx | 反向代理 `/api/` → app，静态直服 `/uploads/`、`/results/`、前端页面，监听 80 |
| MySQL | 复用宿主机 MySQL 8.4.11，库 `faces`，用户 `face`，utf8mb4 |

底库：`face_samples.embedding` 存 128 维 float32（512 字节）；识别时按人求**类中心（centroid）**后做余弦比对。

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
│   ├── engine.py           # 检测 / 特征 / 中文标注
│   ├── database.py         # SQLAlchemy 表定义
│   ├── seed_gallery.py     # 扫描 data/gallery 重建底库
│   ├── calibrate.py        # 留一法看分数分布、建议阈值
│   ├── experiment.py       # 聚合方式 × TTA 对比实验
│   ├── experiment2.py      # 阈值 + margin 决策规则搜索
│   ├── verify.py           # 端到端自检（21 项断言）
│   └── fonts/simhei.ttf
├── web/index.html          # 单页前端（4 个 tab）
├── nginx/default.conf
└── data/
    ├── gallery/<人名>/*.jpg   # 底库原图（27 张 / 3 人）
    ├── uploads/YYYY-MM-DD/    # 上传原图
    └── results/               # 标注结果图
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
sudo docker compose exec -T app python verify.py              # 端到端自检
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

## 六、可调参数（docker-compose.yml → app.environment）

| 变量 | 默认 | 说明 |
|---|---|---|
| `THRESHOLD` | 0.363 | 余弦相似度阈值 |
| `MARGIN` | 0.03 | 与第二名的最大允许差距 |
| `DET_SCORE` | 0.55 | 人脸检测置信度阈值，漏检多则降到 0.45 |
| `MIN_SIDE` | 480 | 短边低于此值放大后再检测 |
| `MAX_SIDE` | 1280 | 长边超过此值缩小，省 CPU |
| `MAX_FILES` | 20 | 单次请求最多图片数 |

## 七、数据库

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

sudo mysqldump --single-transaction faces > ~/faces-$(date +%F).sql   # 备份
```

Navicat：`ssh -L 3307:127.0.0.1:3306 chen@127.0.0.1 -p 2222`，再连 `127.0.0.1:3307`。

## 八、摄像头说明

摄像头代码完整保留，但浏览器要求**安全上下文**才允许 `getUserMedia`：

- `http://localhost:8080`（SSH 隧道）→ 可用
- `http://<IP>` → 浏览器禁用，页面给出明确提示
- 设备无摄像头 → 提示「未检测到摄像头设备」，改用「上传图片识别」

## 九、排障

| 现象 | 处理 |
|---|---|
| 页面 502 | `docker compose ps` 看 app 是否 healthy；`docker compose logs app` 查连库 |
| 连不上 MySQL | `mysql -uface -p -h127.0.0.1 faces` 手动试密码；app 必须用 host 网络才能访问 127.0.0.1:3306 |
| 中文显示方块 | 确认 `app/fonts/simhei.ttf` 在镜像内 |
| 某人总识别不出 | 底库图太少/太糊，在网页「人脸库管理」补 3 张以上清晰正脸照 |
| 结果图 404 | 确认 `data/uploads`、`data/results` 存在且 0755 |
