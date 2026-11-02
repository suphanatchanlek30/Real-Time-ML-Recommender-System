# Runbook: รันระบบออนไลน์ (Kafka + Redis + API)

ต้องเทรนโมเดลก่อน (ดู [runbook-training.md](runbook-training.md)) ให้มี alias `champion` ใน MLflow และมี `data/features/` ครบ

## 0. เตรียม (ครั้งเดียว)

```bash
cd "/Users/suphanatchanlek/Desktop/Projecst ML/ML-Recommender-System"
source .venv/bin/activate        # Python 3.11
make install-all                 # เพิ่ม fastapi, redis, kafka client, fakeredis ฯลฯ
```

เปิด **Docker Desktop** ให้เรียบร้อยก่อน (เช็กด้วย `docker info` ต้องไม่ error)

## ตั้งค่าด้วย `.env`

ค่าตั้งค่าทั้งหมด (พอร์ต, image, ชื่อ topic, timeout ของ Redis, snapshot, token ของ `/admin/reload`, อัตรา replay) อยู่ในไฟล์ `.env` ที่เดียว ไม่ได้เขียนไว้ใน Dockerfile หรือโค้ด

```bash
make env                         # สร้าง .env จาก .env.example (ถ้ายังไม่มี) ทำให้อัตโนมัติตอน make up
```

แก้ค่าใน `.env` แล้วรัน `make up` ใหม่ได้เลย ไม่ต้อง rebuild image (ยกเว้นเปลี่ยนโค้ด) ไฟล์ `.env` ไม่ถูก commit (เก็บ token ได้) ส่วน `.env.example` เป็นแม่แบบที่ commit ไว้

| กลุ่ม | ตัวแปร | ใช้ทำอะไร |
| --- | --- | --- |
| พอร์ตบนเครื่อง | `API_PORT`, `REDIS_PORT`, `KAFKA_EXTERNAL_PORT` | เปลี่ยนเมื่อพอร์ตชนกับโปรแกรมอื่น |
| Image | `KAFKA_IMAGE`, `REDIS_IMAGE` | เลือกเวอร์ชัน |
| Redis | `REDIS_HOST`, `REDIS_DB`, `REDIS_TIMEOUT_S` | ที่อยู่และ timeout (ช้ากว่านี้ API ตอบแบบไม่มีฟีเจอร์ผู้ใช้) |
| Kafka | `KAFKA_TOPIC`, `KAFKA_GROUP`, `KAFKA_PARTITIONS`, `KAFKA_REPLAY_RATE` | topic, consumer group, จำนวน partition, events ต่อวินาทีตอน replay |
| โมเดล/ฟีเจอร์ | `MODEL_DIR`, `FEATURES_DIR`, `FEATURES_SNAPSHOT`, `DATA_DIR` | path บนเครื่อง และ snapshot ที่ API ใช้ |
| API | `ADMIN_TOKEN` | ถ้าตั้งค่า `POST /admin/reload` ต้องส่งหัว `X-Admin-Token` |
| MLflow | `MLFLOW_TRACKING_URI` | ที่เก็บผลเทรน (ค่าเริ่มต้นเป็นไฟล์ SQLite) |

โค้ด Python บนเครื่อง (เช่น `make load-redis`, `make smoke`) อ่าน `.env` เอง ส่วนใน container ค่าเหล่านี้ถูกส่งต่อโดย `docker-compose.yml` ผ่าน `${ชื่อตัวแปร}` สิ่งที่ยังอยู่ใน compose คือการเชื่อมต่อภายใน container ที่เปลี่ยนไม่ได้ (ชื่อ host `redis`/`kafka` และจุด mount ของโมเดลกับฟีเจอร์) ตัวแปรที่ตั้งไว้ใน shell จะชนะค่าใน `.env`

## 1. ดึงโมเดลที่ champion

```bash
make pull-model                  # -> models/champion/ (als.npz, ranker.txt, schema.json, ...)
```

## 2. เปิดระบบ

```bash
make up                          # build image และเปิด kafka, redis, api, consumer (ครั้งแรก ~2 นาที)
curl localhost:8000/health       # {"status":"ok", "model_version":"1", ...} (พอร์ตตาม API_PORT ใน .env)
```

## 3. โหลดข้อมูลตั้งต้นและ replay event

```bash
make replay                      # ~3 นาที
```

ทำตามลำดับ: หยุด consumer → ล้าง Redis แล้วโหลด state ของผู้ใช้ 1.28 ล้านคน (~30 วินาที) → สร้าง topic ใหม่ → เปิด consumer → ส่ง 233,717 events ที่ 2,000/วินาที

ดูความคืบหน้าในอีกแท็บ: `make logs` (consumer จะพิมพ์ `processed {'ok': ...}` ทุก 10 วินาที)

## 4. ลองเรียก API

```bash
curl "localhost:8000/recommend?user_id=9145&k=10"
```

ตัวอย่างผลลัพธ์:

```json
{"items":[{"item_id":350790,"score":1.23}, ...],
 "model_version":"1","source":"ranker","retrieval":"als","degraded":false}
```

| ฟิลด์ | ความหมาย |
| --- | --- |
| `source` | `ranker` = Ranker เรียงให้ / `popular_fallback` = Ranker พัง ส่ง popular list แทน |
| `retrieval` | `als` = candidate จาก ALS / `popular` = ผู้ใช้ที่ ALS ไม่รู้จัก |
| `degraded` | `true` = Redis ช้าหรือล่ม จึงไม่มีฟีเจอร์ผู้ใช้ |

**ดูว่า replay เปลี่ยนผลจริง:** เรียก `curl "localhost:8000/recommend?user_id=9145&k=10"` ก่อนรัน `make replay` (หลังโหลด state ตั้งต้น) แล้วเรียกอีกครั้งหลัง replay จบ ลำดับสินค้าจะต่างกันเล็กน้อย

## 5. เช็กว่าทุกอย่างทำงาน

```bash
make smoke                       # ยิง event เข้า Kafka จริง แล้วเช็กว่าไปถึง Redis และ API
curl localhost:8000/metrics | head -20
curl localhost:8000/health
```

`event_clock` ใน `/health` คือเวลาของ event ล่าสุดที่ประมวลผล (หลัง replay จะเป็น 2015-09-18)

## 6. เปลี่ยนหรือถอยโมเดลโดยไม่ rebuild

```bash
# 1) ย้าย alias champion ใน MLflow ไปเวอร์ชันที่ต้องการ (make mlflow-ui)
make pull-model
curl -X POST localhost:8000/admin/reload
# -> {"previous_version":"1","model_version":"2"}
```

ถ้า bundle ใหม่ใช้ไม่ได้ (ลำดับฟีเจอร์ไม่ตรง หรือ snapshot ไม่ตรง) API ตอบ 409 และใช้โมเดลเดิมต่อ

## 7. ปิดระบบ

```bash
make down
```

## ถ้าติดปัญหา

| อาการ | วิธีแก้ |
| --- | --- |
| `Cannot connect to the Docker daemon` | เปิด Docker Desktop แล้วรอจนไอคอนนิ่ง |
| `variable is not set. Defaulting to a blank string` | ไม่มี `.env` หรือขาดตัวแปร: รัน `make env` และเทียบกับ `.env.example` |
| `/health` ตอบ `no_model` | ยังไม่ได้ `make pull-model` หรือ `models/champion/` ไม่ครบ ดู `docker compose logs api` |
| `reload failed ... snapshot` | โมเดลกับ `data/features/` ไม่ใช่ชุดเดียวกัน รัน `make features` และ `make pull-model` ใหม่ |
| `/health` ตอบ `degraded` | Redis ไม่พร้อม: `docker compose ps` แล้วลอง `docker compose restart redis` จากนั้น `make load-redis` |
| consumer ไม่ประมวลผล | ดู `make logs` ถ้า topic เพิ่งถูกสร้างใหม่ให้รัน `docker compose restart consumer` |
| อยากเริ่มข้อมูลสดใหม่ทั้งหมด | `make replay` (ทำให้ใหม่ทุกอย่างเอง) |
| RAM ไม่พอ | ระบบใช้ ~2.7 GB (Kafka ~1 GB, Redis ~0.8 GB, API ~0.7 GB) เพิ่ม RAM ให้ Docker Desktop ที่ Settings > Resources |
