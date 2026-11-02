# Runbook: เทรนโมเดลบน Mac (copy วางทีละบล็อก)

รันทุกคำสั่งใน Terminal ที่โฟลเดอร์โปรเจกต์ เวลาที่ใส่ไว้วัดจริงบน MacBook (M5) ครั้งเดียว ของคุณอาจต่างเล็กน้อย

## 0. เตรียม environment (ทำครั้งเดียว)

`.venv` เดิมใช้ Python 3.15 ซึ่งใหม่เกินไปสำหรับไลบรารีเทรน ให้สร้างใหม่ด้วย Python 3.11

```bash
cd "/Users/suphanatchanlek/Desktop/Projecst ML/ML-Recommender-System"
deactivate 2>/dev/null; rm -rf .venv
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version        # ต้องเป็น Python 3.11.x
pip install --upgrade pip
make install-all        # ติดตั้ง polars, pandera, implicit, lightgbm, mlflow, fastapi, pytest ฯลฯ (~2 นาที)
```

ถ้ายังไม่มี Python 3.11: `brew install python@3.11`

## 1. เช็กว่าทุกอย่างพร้อม

```bash
make test               # ต้องผ่านทั้งหมด (66 tests, อีก 4 ข้ามเพราะต้องมี Docker)
ls data/raw             # ต้องเห็น events.csv, item_properties_part*.csv, category_tree.csv
```

## 2. ข้อมูลและฟีเจอร์ (ถ้าเคยรันแล้วข้ามได้)

```bash
make etl                # ~3 วินาที   -> data/processed/*.parquet
make features           # ~15 วินาที  -> data/features/{rank_1..rank_4,val,test}/
```

## 3. จูน ALS (ไม่บังคับ แต่แนะนำ)

ลอง 12 ชุด hyperparameter แล้วเก็บชุดที่ดีที่สุดไว้ให้ `make train` ใช้ (factors 128 ใช้เวลาชุดละ ~1 นาที)

```bash
make tune-als           # ~6 นาที   -> reports/als_tuning.md, reports/als_best.json
cat reports/als_tuning.md
```

## 4. เทรน ALS + Ranker และลงทะเบียนใน MLflow

```bash
make train              # ~2 นาที (ถ้ารัน tune-als แล้วได้ factors 128 คาดว่า ~7 นาที เพราะต้องเทรน ALS 6 ครั้ง)
```

ระหว่างรันจะเห็น log ตามลำดับ: สร้างข้อมูลเทรน Ranker จาก 4 snapshot → เทรน Ranker → ตารางผลบน Val → บรรทัด `gate vs ...: PASS/FAIL`

- **PASS:** ลงทะเบียนเวอร์ชันใหม่ใน MLflow Registry แล้วย้าย alias `champion` ไปที่เวอร์ชันนั้น
- **FAIL:** log ผลไว้ใน MLflow แต่ไม่ลงทะเบียน (โมเดลใหม่ต้องดีกว่า champion เดิมที่ NDCG@10)
- ผลทั้งหมดเขียนที่ `reports/train_val.md` ด้วย

อยากลองโดยไม่แตะ MLflow: `python -m src.training.train --no-mlflow`

## 5. ดูผลใน MLflow

```bash
make mlflow-ui          # เปิด http://127.0.0.1:5000 (กด Ctrl+C เพื่อปิด)
```

- แท็บ **Experiments → recsys**: ดู params, metrics, tags (git commit, hash ของข้อมูล) ของแต่ละ run
- แท็บ **Models → recsys-bundle**: ดูเวอร์ชันและ alias `champion`
- ไฟล์ของโมเดล (`als.npz`, `ranker.txt`, `schema.json`, `popular.json`) อยู่ใน artifacts ของ run

## ถ้าติดปัญหา

| อาการ | วิธีแก้ |
| --- | --- |
| `No module named 'implicit'` / `lightgbm` | ยังไม่ได้ `source .venv/bin/activate` หรือ `make install-train` |
| `pip install implicit` ล้มเหลว | เช็กว่า `python --version` เป็น 3.11 (ไม่ใช่ 3.15) |
| `FileNotFoundError ... events.parquet` | รัน `make etl` ก่อน |
| `FileNotFoundError ... data/features/rank_1` | รัน `make features` (ต้องรันใหม่ถ้าเคยรันก่อนเพิ่ม snapshot หลายช่วง) |
| อยากเริ่ม MLflow ใหม่ทั้งหมด | `rm -rf mlflow.db mlruns` แล้วรัน `make train` ใหม่ |
| รัน `make train` ซ้ำแล้ว `gate ... FAIL` | ปกติ ผลเหมือนเดิมเป๊ะ จึง "ไม่ได้ดีกว่า" champion ต้องปรับอะไรสักอย่างก่อน |
