# Real-Time ML Recommender System

> Two-stage recommendation pipeline: from raw events to a low-latency serving API

ระบบแนะนำสินค้าแบบครบวงจร (end-to-end) สำหรับร้านค้าออนไลน์ ตั้งแต่จัดการข้อมูล ออกแบบฟีเจอร์ เทรนโมเดล ไปจนถึง API ที่แนะนำสินค้าให้ผู้ใช้แต่ละคนแบบเรียลไทม์ โดยฟีเจอร์ถูกอัปเดตสดจาก Kafka ไปยัง Redis และมี Prometheus/Grafana กับ k6 วัดผลจริง ออกแบบให้เหมือนงานของ ML Engineer ไม่ใช่แค่ notebook

| | ผลที่ได้ |
| --- | --- |
| **คุณภาพโมเดล** (Validation, 1,941 ผู้ใช้) | NDCG@10 **0.033** เทียบกับ baseline "สินค้ายอดนิยม" 0.010 (ผู้ใช้เก่า 0.132 เทียบกับ 0.009) |
| **ความเร็ว** (100 VUs) | p95 **30 ms**, error 0%, 695 request/วินาที |
| **ความจุ** | รักษา p95 < 100 ms ได้ถึง ~1,360 request/วินาที (ก่อนปรับปรุง ~300) |
| **ความถูกต้อง** | ฟีเจอร์ตอน serve ตรงกับตอนเทรนทุกตัว มี test พิสูจน์ (89 tests) |

## สารบัญ

1. [สถาปัตยกรรม](#1-สถาปัตยกรรม)
2. [Data](#2-data)
3. [เทคนิคและวิธีการ](#3-เทคนิคและวิธีการ)
4. [ผลลัพธ์](#4-ผลลัพธ์)
5. [Quick start](#5-quick-start)
6. [คำสั่งทั้งหมด](#6-คำสั่งทั้งหมด)
7. [การตั้งค่า (.env)](#7-การตั้งค่า-env)
8. [Testing และ CI](#8-testing-และ-ci)
9. [โครงสร้างโปรเจกต์](#9-โครงสร้างโปรเจกต์)
10. [Design decisions และ trade-offs](#10-design-decisions-และ-trade-offs)
11. [ข้อจำกัดและงานต่อยอด](#11-ข้อจำกัดและงานต่อยอด)
12. [เอกสารเพิ่มเติม](#12-เอกสารเพิ่มเติม)

## 1. สถาปัตยกรรม

ระบบมี 2 ฝั่งที่เชื่อมกันด้วย 3 เส้นหลัก:

| เส้นที่เชื่อม | ทำอะไร |
| --- | --- |
| Feature engineering → Redis | โหลดสถานะผู้ใช้จากข้อมูลเทรนเข้า Redis แล้ว Kafka consumer อัปเดตต่อแบบสด ใช้โค้ดคำนวณชุดเดียวกัน (กัน training-serving skew) |
| Model registry → FastAPI | API โหลดโมเดลที่ alias `champion` อัปเกรดหรือถอยเวอร์ชันได้โดยไม่ต้อง rebuild image |
| FastAPI → Prometheus | เปิด `/metrics` ให้เก็บค่า ทำให้ผล k6 และ dashboard ยืนยันตัวเลขได้ |

### Offline: ข้อมูลไปจนถึงโมเดล

events ดิบ → ETL → Features → ALS → Ranker → Evaluation gate → Model registry

![Offline pipeline](docs/diagrams/1-offline-pipeline.drawio.svg)

### Online: Streaming, Serving, Monitoring

สาย streaming (Kafka → consumer → Redis) อัปเดตฟีเจอร์ล่าสุด และสาย serving (FastAPI) ตอบ request พร้อมวัดผลด้วย Prometheus/Grafana

![Online serving](docs/diagrams/2-online-serving.drawio.svg)

### ภาพรวมทั้งระบบ

![End-to-end](docs/diagrams/3-end-to-end.drawio.svg)

> ภาพทั้งสามเป็นแบบที่ออกแบบไว้ก่อนเริ่มทำ ตอน implement มีบางจุดต่างไป (เช่น ไม่ใช้ TTL ใน Redis, MLflow รันบนเครื่องแทนใน compose) เหตุผลอยู่ใน [Design decisions](#10-design-decisions-และ-trade-offs)

## 2. Data

**แหล่งที่มา:** [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (Kaggle) เป็น event log จริงของร้านออนไลน์ มี timestamp จึง replay เป็น stream ได้

| ไฟล์ | ใช้ทำอะไร |
| --- | --- |
| `events.csv` | ข้อมูลหลัก (visitor, item, event, เวลา) 2,756,101 แถว ผู้ใช้ 1.41 ล้านคน สินค้า 235,061 ชิ้น ช่วง 3 พ.ค. - 18 ก.ย. 2015 (137 วัน) |
| `item_properties_part1/2.csv` | ใช้เฉพาะ `categoryid` และ `available` (ไฟล์รวม ~850 MB ดึงออกมาครั้งเดียวเหลือ 4 MB) |
| `category_tree.csv` | ลำดับชั้นหมวดหมู่ (1,669 หมวด) |

Event มี 3 ชนิด: `view` 96.67%, `addtocart` 2.52%, `transaction` 0.81%

**ข้อค้นพบจาก EDA ที่กำหนดการออกแบบ** ([notebook](notebooks/01_eda.ipynb)):

| พบ | ผลต่อการออกแบบ |
| --- | --- |
| 71% ของผู้ใช้มี event เดียว, 78% ของ session มี event เดียว | ALS เรียนเฉพาะผู้ใช้ที่มี event >= 2 ครั้ง ผู้ใช้ใหม่ต้องมีทางสำรอง |
| ผู้ใช้ที่ event สูงผิดปกติ **ไม่ใช่บอทชัดเจน** (ซื้อบ่อยกว่าปกติ ~6 เท่า) | ไม่ตัดที่ percentile ตรงๆ ใช้กฎที่ไม่ทิ้งผู้ซื้อจริง |
| ผู้ใช้ใน Val/Test เป็นผู้ใช้เก่าเพียง ~15% | ต้องรายงานผลแยก warm / cold |
| สินค้า top 1% = 23% ของ events | baseline "ยอดนิยม" จะแข็ง ต้องเอาชนะให้ได้ |
| ความยาว session ทำนายการซื้อได้ (session 1 event ซื้อ 0.04%, 10+ events ซื้อ 19%) | ใช้ session เป็นฟีเจอร์ |

**การทำความสะอาด (ETL):** ลบแถวซ้ำ 460 แถว, ตัดผู้ใช้ผิดปกติ 651 คน (ที่ event สูงแต่ไม่เคยใส่ตะกร้า/ซื้อ หรือสูงเกิน p99.99), ตรวจ schema ด้วย Pandera ทุกครั้งที่อ่านและเขียน Parquet เหลือ 2,595,796 events

**การแบ่งข้อมูล (ตามเวลา ห้ามสุ่ม):**

```text
|-------------- Train --------------|--- Val ---|--- Test ---|
3 พ.ค.                            4 ก.ย.      11 ก.ย.      18 ก.ย.
2,362,079 events                  125,044     108,673
```

## 3. เทคนิคและวิธีการ

### 3.1 กัน data leakage

- **แบ่งตามเวลา** เพราะระบบจริงต้องทำนายอนาคตจากอดีต ถ้าสุ่มแบ่ง โมเดลจะเห็น event ของวันหลังตอนเทรน
- **ฟีเจอร์เป็น point-in-time:** ทุกฟังก์ชันรับ `as_of` และใช้เฉพาะ event ที่ `ts < as_of` มี test ที่เติม event อนาคตเข้าไปแล้วผลต้องไม่เปลี่ยน (ผมลองแก้โค้ดให้รั่วแล้ว test ล้มจริง)
- ค่าของ `category`/`available` ใช้ค่า "ที่เป็นอยู่ ณ เวลานั้น" ไม่ใช่ค่าล่าสุดทั้งไฟล์
- **Test set ถูกกันไว้ ยังไม่ได้ประเมินโดยตั้งใจ** จะดูครั้งเดียวกับโมเดลสุดท้าย

### 3.2 วิธีประเมิน

- Ground truth = สินค้าที่ผู้ใช้ใส่ตะกร้าหรือซื้อในช่วงนั้น (view ไม่นับ)
- ตัวชี้วัด: Recall@K, Hit@K, NDCG@K, MAP@K, Coverage@K ทุกค่ามี **ช่วงความเชื่อมั่น 95%** (bootstrap ข้ามผู้ใช้) เพราะผู้ใช้ที่วัดได้มีแค่ ~1.9K คน
- รายงานแยก **warm** (เคยอยู่ใน Train) กับ **cold** (ไม่เคยเห็น)
- เทียบกับ baseline: `random`, `popular_all`, `popular_7d` (ยอดนิยมที่ถ่วงน้ำหนัก view 1 / addtocart 3 / transaction 5)

### 3.3 ฟีเจอร์ 38 ตัว ([src/features/](src/features/))

| กลุ่ม | จำนวน | ตัวอย่าง |
| --- | --- | --- |
| user | 11 | จำนวน view/cart/buy, sessions, recency, decayed interest |
| item | 16 | popularity 1/7/28 วัน, trend, unique users, อายุ, conversion แบบ smoothed, category, availability |
| category | 3 | popularity ของหมวด |
| pair (user × item) | 5 | ผู้ใช้เคย view/cart/buy สินค้านี้กี่ครั้ง, ห่างกี่วัน, สัดส่วนหมวดที่ผู้ใช้สนใจ |
| candidate | 3 | คะแนน ALS, อันดับที่ดึงมา, ดึงจาก ALS หรือสำรอง |

**ให้ offline = online:** ฟีเจอร์ผู้ใช้แยกเป็น *state* (นับ, เวลาแรก/ล่าสุด, ผลรวมที่เสื่อมตามเวลา, 20 สินค้าล่าสุด ซึ่งอัปเดตทีละ event ได้) กับ *derive* (recency, ratio คำนวณตอนใช้) เพื่อไม่เก็บค่าที่เก่าทันทีที่เขียน รายการฟีเจอร์ทั้งหมดอยู่ที่เดียว ([spec.py](src/features/spec.py)) แล้วสร้าง Pandera schema และ `schema.json` ที่เซฟคู่โมเดลอัตโนมัติ ตอน serve จะปฏิเสธโมเดลที่ลำดับฟีเจอร์ไม่ตรง

### 3.4 โมเดลสองขั้น

1. **ALS** (`implicit`) ดึง candidate 200 ตัวต่อผู้ใช้ จาก interaction ที่ถ่วงน้ำหนัก scoring ตอน serve คือ dot product ด้วย numpy (ไม่ต้องมี `implicit`) จูน hyperparameter 12 ชุด ได้ factors 128 / regularization 0.01 / alpha 40
2. **Ranker** (LightGBM LambdaRank) เรียง 200 ตัวนั้นใหม่ด้วยฟีเจอร์ 38 ตัว เทรนบน 4 snapshot ต่อเนื่อง (ฟีเจอร์จากอดีต, label จาก 7 วันถัดมา) ผู้ใช้ที่ ALS ไม่รู้จักได้ candidate เป็นสินค้ายอดนิยม 7 วันล่าสุด
3. **Evaluation gate:** โมเดลใหม่ต้องมี NDCG@10 ดีกว่า champion เดิม (หรือ baseline ถ้ายังไม่มี) และ Recall@200 ไม่แย่ลง จึงลงทะเบียนใน MLflow Registry และย้าย alias `champion` ส่วน bundle ที่ log ไว้ประกอบด้วยโมเดล, `schema.json`, รายการ popular พร้อม tag ของ git commit และ hash ของข้อมูล

**ทำไมเป็นสองขั้น:** ดึง candidate ด้วยโมเดลเบา (ALS) จากสินค้าราว 160K ตัว แล้วใช้โมเดลที่ฉลาดกว่า (Ranker ที่ใช้ฟีเจอร์ตาราง) เรียงเฉพาะ 200 ตัว ได้ทั้งความเร็วและคุณภาพ และ Ranker ใช้ฟีเจอร์ที่ ALS ใช้ไม่ได้ เช่น "ผู้ใช้เคยใส่ตะกร้าสินค้านี้"

### 3.5 ระบบออนไลน์

```text
GET /recommend?user_id=..  ->  Redis (1 round trip: state, per-item counts, category counts, clock)
                           ->  200 candidates (ALS dot product / popular list)
                           ->  38 features  ->  LightGBM  ->  top-k + model_version + source
```

- **Event-time clock:** ข้อมูลเป็นปี 2015 ถ้าใช้เวลาเครื่อง recency จะเป็น ~4,000 วันและค่าที่เสื่อมตามเวลาเป็น 0 ทุกคน จึงเก็บเวลาของ event ล่าสุดไว้ใน Redis แล้วคำนวณฟีเจอร์ ณ เวลานั้น
- **เริ่มจากจุดเดียวกับ champion:** โหลด state ณ `val_start` เข้า Redis (1.28 ล้านผู้ใช้ ใน 28 วินาที ใช้ RAM ~720 MB) แล้ว replay เฉพาะ event หลังจากนั้น `make replay` ล้าง Redis ก่อนทุกครั้งจึงไม่นับซ้ำ
- **Streaming ที่ไม่นับซ้ำ:** Kafka ส่งอย่างน้อยหนึ่งครั้ง (หลัง crash จะอ่าน batch เดิมซ้ำ) consumer จึงเก็บ offset ล่าสุดที่ใช้ไว้ใน Redis ใน transaction เดียวกับการอัปเดต state ตอนทดสอบ kill consumer กลางทาง ก่อนแก้นับ event ซ้ำกับผู้ใช้ 343 คน หลังแก้ตรวจทั้ง 1.4 ล้านคนแล้วผิด 0 คน
- **ทางสำรอง:** Redis ช้า/ล่ม (timeout 50 ms) → ตอบโดยไม่มีฟีเจอร์ผู้ใช้ (`degraded: true`) · Ranker พัง → ตอบ popular list (`source: popular_fallback`) · ผู้ใช้ใหม่ → candidate จาก popular
- **Hot reload:** `POST /admin/reload` สร้างโมเดลใหม่ให้เสร็จและตรวจก่อนสลับ ถ้า bundle เสีย (ลำดับฟีเจอร์ไม่ตรง หรือ snapshot ไม่ตรง) ตอบ 409 และใช้โมเดลเดิมต่อ ถอยเวอร์ชันได้โดยย้าย alias แล้ว `make pull-model`
- **ใช้ผลซ้ำอย่างปลอดภัย:** ผู้ใช้ใหม่ที่ไม่มีประวัติทุกคนได้ candidate และฟีเจอร์ชุดเดียวกัน จึงคำนวณรายการเรียงลำดับครั้งเดียวต่อโมเดล (test ยืนยันว่าคำตอบเท่าเดิมทุก k)

### 3.6 พิสูจน์ว่าถูกต้อง

- **offline = online** ทดสอบบนข้อมูลสังเคราะห์ (ฟีเจอร์จาก Redis ที่เติมด้วย stream เท่ากับฟีเจอร์ offline) และบนโมเดล champion จริงกับผู้ใช้ 120 คน (ฟีเจอร์และลำดับที่ API แนะนำเท่ากับเส้นทางตอนเทรนทุกตัว)
- **ทดสอบ mutation:** แก้สูตรให้ผิดโดยตั้งใจแล้วตรวจว่า test ล้ม (เช่น สูตร decay, สัดส่วนหมวด, การกรองอนาคต)
- **ทดสอบความทนทาน:** kill consumer กลางการ replay, reload ด้วย bundle เสีย, เปลี่ยนทุกค่าใน `.env`

### 3.7 วิธี Load test และ Monitoring

- **Prometheus** เก็บ latency (histogram), RPS, error, แหล่งที่มาของคำตอบ, เวลาอ่าน Redis, CPU/RAM, Kafka lag · **Grafana** มี dashboard ที่สร้างอัตโนมัติจากไฟล์
- **k6 สามแบบ:** smoke (1 VU) · load (ไต่ถึง 100 VUs แล้วค้าง, เกณฑ์ p95 < 100 ms และ error < 1%) · stress (เพิ่มอัตรา request ไปเรื่อยๆ แบบ open model เพื่อให้เห็นอาการล้นจริง)
- **user id สมจริง:** 15% ผู้ใช้ที่ ALS รู้จัก, 35% มีประวัติใน Redis, 50% ผู้ใช้ใหม่ ไม่ใช่เลขสุ่มล้วน
- **วิธีปรับปรุงตามหลักฐาน:** วัด baseline → ดูกราฟ/profile หาคอขวด → แก้ → วัดซ้ำด้วยชุดทดสอบเดิม ขั้นตอนและกราฟทุกรอบอยู่ใน [docs/loadtest.md](docs/loadtest.md)

## 4. ผลลัพธ์

### 4.1 คุณภาพโมเดล (Validation, 1,941 ผู้ใช้: warm 289, cold 1,652)

| Model | NDCG@10 | Recall@10 | Recall@200 |
| --- | --- | --- | --- |
| random | 0.0002 | 0.0005 | 0.0018 |
| popular_7d (baseline) | 0.0102 | 0.0191 | 0.1032 |
| ALS | 0.0170 | 0.0295 | 0.1263 |
| **ALS + Ranker** | **0.0333** | **0.0438** | 0.1263 |

| กลุ่ม | NDCG@10: popular_7d → ALS → ALS + Ranker |
| --- | --- |
| ผู้ใช้เก่า (289) | 0.0090 → 0.0550 → **0.1318** |
| ผู้ใช้ใหม่ (1,652) | 0.0104 → 0.0104 → **0.0160** |

- ผู้ใช้ส่วนใหญ่เป็นผู้ใช้ใหม่ ALS จึงไม่ช่วยกลุ่มนี้ ที่ดีขึ้นเล็กน้อยมาจาก Ranker ช่วงความเชื่อมั่นยังทับกับ baseline บางส่วน
- ผู้ใช้เก่าดีขึ้นมาก ส่วนใหญ่มาจากฟีเจอร์ "เคยดู/ใส่ตะกร้าสินค้านี้" ซึ่งสะท้อนพฤติกรรมกลับมาซื้อซ้ำ
- Recall@200 ของ ALS และ ALS + Ranker เท่ากัน เพราะ Ranker แค่เรียง 200 ตัวเดิมใหม่
- ตารางเต็มพร้อมช่วงความเชื่อมั่น: [reports/train_val.md](reports/train_val.md)

### 4.2 Load test (k6 + Prometheus)

| | ก่อน | หลังปรับปรุง |
| --- | --- | --- |
| Load 100 VUs: p95 | 166 ms (**ไม่ผ่านเกณฑ์**) | **30 ms** (ผ่าน) |
| Load 100 VUs: p50 / p99 | 93 / 195 ms | 2.8 / 70 ms |
| request / วินาที | 401 | **695** |
| Stress: อัตราที่รักษา p95 < 100 ms | ~300 / วินาที | **~1,360 / วินาที** |
| CPU ของ API ตอน load test | 9.6 คอร์ | 1.4 คอร์ |
| Error | 0% | 0% |

![ก่อนและหลังปรับปรุง](docs/loadtest/comparison.png)

**ก่อนปรับปรุง** (100 VUs): กราฟแสดง API กิน CPU ~9.4 คอร์เพื่อรับ ~500 request/วินาที (ควรใช้ ~1 คอร์) เพราะ thread ของ numpy/LightGBM แย่งกัน และ p95 เกินเป้า

![baseline load](docs/loadtest/baseline-load.png)

**หลังปรับปรุง** (100 VUs): p95 อยู่ที่ 30 ms ต่ำกว่าเป้า

![after load](docs/loadtest/stage2-code-load.png)

**Stress หลังปรับปรุง:** ระบบนิ่งจนถึง ~1,360 request/วินาที แล้วคิวสะสม (requests in flight พุ่ง) และ latency ขึ้นเป็นวินาที โดยไม่มี error เพราะ request เข้าคิวไม่ถูกปฏิเสธ

![after stress](docs/loadtest/stage2-code-stress.png)

**ปรับปรุงอะไรไป:** (1) จำกัด thread ของ numpy/LightGBM เหลือ 1 CPU ลด 5 เท่าแต่ throughput แทบไม่ขึ้น (2) profile ต่อ request ชี้ว่าการสร้าง DataFrame polars เพื่อคำนวณฟีเจอร์กิน ~0.94 ms จึงเปลี่ยนเป็น Python ล้วน (test เทียบกับโค้ด offline) พร้อมปรับการหา pair features, middleware และใช้รายการเรียงลำดับของผู้ใช้ใหม่ซ้ำ ผลคือ 401 → 695 request/วินาที

> ตัวเลข load test วัดบนเครื่องเดียว (k6, API, Redis, Kafka อยู่ใน Docker เดียวกัน) และรอบเดียวต่อชุดการตั้งค่า เหมาะกับการเทียบก่อน/หลัง ไม่ใช่ความจุบน production รายละเอียดและข้อจำกัดอยู่ใน [docs/loadtest.md](docs/loadtest.md)

### 4.3 Dashboard

เปิด Grafana ที่ <http://localhost:3000> (พอร์ตตั้งใน `.env`) หน้าแรกคือ **Recommender: service overview** มี RPS, error rate, p50/p95/p99 (เส้นเป้า 100 ms), แหล่งที่มาของคำตอบ, เวลาอ่าน Redis, CPU/RAM ของ API, Kafka lag และเส้นฝั่ง k6 ให้เปิดดูระหว่างรัน `make loadtest` (ภาพที่แสดงในหัวข้อ 4.2 วาดจากข้อมูลชุดเดียวกันใน Prometheus)

## 5. Quick start

**ต้องมี:** Python 3.11 (ที่ทดสอบ), Docker Desktop ที่จัดสรร RAM ได้ 6 GB ขึ้นไป (ระบบใช้ราว 3 GB), [Kaggle API key](https://www.kaggle.com/settings) (`~/.kaggle/kaggle.json`)

```bash
# 1) เตรียม environment และข้อมูล
python3.11 -m venv .venv && source .venv/bin/activate
make install-all && make env
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip

# 2) ข้อมูล -> ฟีเจอร์ -> โมเดล (รวม ~3-10 นาที)
make etl && make features
make train                       # (ไม่บังคับ: make tune-als ก่อน ~6 นาที)

# 3) เปิดระบบออนไลน์แล้วลองเรียก
make pull-model && make up && make replay
curl "localhost:8000/recommend?user_id=9145&k=10"
```

ตัวอย่างคำตอบ:

```json
{"items":[{"item_id":350790,"score":1.23}, ...],
 "model_version":"1","source":"ranker","retrieval":"als","degraded":false}
```

`source` = `ranker` (Ranker เรียงให้) หรือ `popular_fallback` · `retrieval` = `als` หรือ `popular` · `degraded` = `true` เมื่อ Redis ช้า/ล่ม

ต่อจากนั้น: `make smoke` (ตรวจทั้งระบบ), `make loadtest-users && make loadtest` (load test), `make mlflow-ui` (ดูผลการเทรน) แล้วปิดด้วย `make down` คำอธิบายทีละขั้นอยู่ใน [runbook](#12-เอกสารเพิ่มเติม) ของแต่ละส่วน

## 6. คำสั่งทั้งหมด

| กลุ่ม | คำสั่ง | ทำอะไร |
| --- | --- | --- |
| ติดตั้ง | `make install-all` / `install-train` / `install` | ติดตั้งไลบรารีทั้งหมด / เฉพาะเทรน / เฉพาะพื้นฐาน |
| | `make env` | สร้าง `.env` จากแม่แบบ |
| ข้อมูล | `make etl` | ทำความสะอาด ตรวจ schema เขียน Parquet และแบ่ง Train/Val/Test |
| | `make features` | สร้างตารางฟีเจอร์ของทุก snapshot |
| โมเดล | `make eval-baseline` | วัด baseline บน Validation |
| | `make tune-als` | จูน hyperparameter ของ ALS |
| | `make train` | เทรน ALS + Ranker ประเมิน และลงทะเบียน champion |
| | `make mlflow-ui` | เปิด MLflow UI |
| ออนไลน์ | `make pull-model` | ดึง bundle ของ champion ลง `models/champion/` |
| | `make up` / `make down` | เปิด / ปิดทั้ง stack |
| | `make load-redis` | โหลด state ตั้งต้นเข้า Redis |
| | `make replay` | ล้าง Redis และ topic แล้ว replay event ผ่าน Kafka |
| | `make logs` | ดู log ของ API และ consumer |
| ทดสอบ | `make test` / `make lint` / `make format` | pytest / ruff |
| | `make smoke` | ทดสอบทั้งระบบที่รันอยู่ (Kafka → Redis → API) |
| Load test | `make loadtest-users` | สร้างชุด user id ที่เหมือนทราฟฟิกจริง |
| | `make loadtest-smoke` / `loadtest` / `loadtest-stress` | k6 สามแบบ |

## 7. การตั้งค่า (.env)

ค่าตั้งค่าทั้งหมดอยู่ใน `.env` ที่เดียว (คัดลอกจาก [.env.example](.env.example) ด้วย `make env`) ไม่ถูก commit จึงเก็บ token ได้ Dockerfile ไม่มีค่าตั้งค่า docker-compose ส่งค่าเข้า container ผ่าน `${...}` และโค้ด Python บนเครื่องอ่านไฟล์นี้เอง

| กลุ่ม | ตัวแปร |
| --- | --- |
| พอร์ตบนเครื่อง | `API_PORT`, `REDIS_PORT`, `KAFKA_EXTERNAL_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT` |
| Docker image | `KAFKA_IMAGE`, `REDIS_IMAGE`, `PROMETHEUS_IMAGE`, `GRAFANA_IMAGE`, `K6_IMAGE` |
| Redis | `REDIS_HOST`, `REDIS_DB`, `REDIS_TIMEOUT_S` |
| Kafka | `KAFKA_TOPIC`, `KAFKA_GROUP`, `KAFKA_PARTITIONS`, `KAFKA_REPLAY_RATE` |
| โมเดล/ฟีเจอร์ | `MODEL_DIR`, `FEATURES_DIR`, `FEATURES_SNAPSHOT`, `DATA_DIR` |
| API | `ADMIN_TOKEN`, `COMPUTE_THREADS` |
| อื่นๆ | `K6_SCENARIO`, `GRAFANA_ADMIN_PASSWORD`, `MLFLOW_TRACKING_URI` |

## 8. Testing และ CI

- **89 tests** (`make test`) ครอบคลุม ETL, ตัวชี้วัด, ฟีเจอร์ (รวม test กัน leakage), offline = online, API, consumer, รายงาน load test และการตั้งค่า อีก 4 tests (`make smoke`) ต้องมี stack รันอยู่
- test ที่ต้องใช้ข้อมูลจริงหรือไลบรารีเสริมจะข้ามเอง ทำให้รันได้แม้ติดตั้งแค่บางส่วน
- **GitHub Actions** ([ci.yml](.github/workflows/ci.yml)): ruff, pytest, ตรวจ `docker-compose.yml`, build image ทั้งสอง ทำงานเมื่อเปิด Pull Request หรือ push เข้า `develop`/`main` (จำลองบนเครื่องใน venv สะอาดไม่มีข้อมูลแล้วผ่าน แต่ยังไม่เคยรันบน GitHub)

## 9. โครงสร้างโปรเจกต์

```text
.
├── src/
│   ├── etl/          # CSV -> clean -> Parquet + Pandera, แบ่งตามเวลา, ดึง item properties
│   ├── features/     # ฟีเจอร์ (spec, user, item, pair, snapshots) ใช้ทั้ง offline และ online
│   ├── evaluation/   # ตัวชี้วัด, ground truth, baseline
│   ├── training/     # ALS, Ranker, evaluation gate, MLflow, จูน ALS
│   ├── serving/      # FastAPI, Redis feature store, recommender, loader, รายงาน load test
│   └── streaming/    # Kafka producer และ consumer
├── tests/            # pytest
├── notebooks/        # EDA
├── loadtest/         # สคริปต์ k6
├── monitoring/       # Prometheus และ Grafana (dashboard สร้างอัตโนมัติ)
├── docker/           # Dockerfile ของ API และ consumer
├── docker-compose.yml, .env.example, Makefile
├── reports/          # ผลประเมินโมเดล, ผลจูน ALS
└── docs/             # progress log, runbook, ผล load test, ภาพออกแบบ (diagrams)
```

## 10. Design decisions และ trade-offs

| การตัดสินใจ | เหตุผล | ราคาที่จ่าย |
| --- | --- | --- |
| โมเดลสองขั้น (ALS + Ranker) | เร็วและแม่น Ranker ใช้ฟีเจอร์ที่ ALS ใช้ไม่ได้ | ต้องดูแลสองโมเดลที่เวอร์ชันตรงกัน (เก็บเป็น bundle เดียว) |
| แบ่งตามเวลา ไม่สุ่ม | จำลองการทำนายอนาคตจริง กัน leakage | ผู้ใช้ใน Val/Test ส่วนใหญ่เป็นผู้ใช้ใหม่ (สมจริงแต่ ALS วัดผลได้ยาก) |
| Kafka แทนเขียน Redis ตรง | แยกส่วนกัน, เรียงตามผู้ใช้, replay ได้, ปรับขนาดได้ | ซับซ้อนขึ้น และต้องกันการนับซ้ำเอง (เก็บ offset ใน Redis) |
| เก็บ hash จำนวนต่อสินค้าใน Redis | ให้ pair features ตอน serve ตรงกับตอนเทรน (แผนเดิมเก็บแค่ 20 ชิ้นล่าสุด) | ใช้ RAM มากขึ้น (~720 MB) |
| ไม่ใช้ TTL ใน Redis (แผนเดิมมี) | state ที่หมดอายุทำให้ฟีเจอร์ online ต่างจากตอนเทรน | Redis โตขึ้นเรื่อยๆ ในระบบจริงต้องมีนโยบายเก็บกวาดแบบอื่น |
| Event-time clock | ข้อมูลเป็นปี 2015 | ต้องส่งเวลาของ event มาด้วยเสมอ |
| ฟีเจอร์ผู้ใช้เป็น Python ล้วนตอน serve | polars แถวเดียวกิน ~1 ms ต่อ request | มีสูตรสองชุด จึงมี test เทียบกับโค้ด offline ทุกครั้ง |
| MLflow บนเครื่อง (SQLite) ไม่อยู่ใน compose | เรียบง่าย ไม่ต้องมี server | ผู้อื่นใช้ร่วมกันไม่ได้ `make pull-model` คัดลอก bundle ไปให้ container |
| ตั้งค่าผ่าน `.env` | ไม่ฝังค่าใน image เปลี่ยนได้โดยไม่ rebuild | ต้องมี `.env` (สร้างจากแม่แบบด้วย `make env`) |

## 11. ข้อจำกัดและงานต่อยอด

**ข้อจำกัด**

- **ผู้ใช้ใหม่ได้ประโยชน์จากโมเดลน้อย** (NDCG@10 0.010 → 0.016 และช่วงความเชื่อมั่นยังทับกับ baseline) ผู้ใช้ที่ ALS ไม่รู้จักได้ candidate จากสินค้ายอดนิยมเท่านั้น สินค้าที่เพิ่งดูใน session จึงไม่ถูกดึงเข้ามา ผลแนะนำเปลี่ยนตาม session แต่เปลี่ยนไม่มาก
- **ข้อมูลประเมินน้อย:** ผู้ใช้ที่วัดได้ ~1.9K คน ตัวเลขจึงแกว่ง และ Val ถูกใช้ทั้งจูนและตัดสิน gate ตัวเลขที่เชื่อได้จริงคือ Test ซึ่งยังไม่ได้แตะ
- **ผลออฟไลน์ไม่เท่าผลออนไลน์** โปรโตคอลประเมินแบบ snapshot ต้นสัปดาห์วัดผลของ session features ไม่ได้ ข้อมูลเป็นปี 2015 ไม่มีโปรไฟล์ผู้ใช้ และสินค้าใหม่ไม่มี vector
- **Load test:** รันบนเครื่องเดียว วัดชุดละรอบเดียว ระบบล้นแล้ว "ช้าลง" ไม่ตอบ error (ไม่มี load shedding) และยังมี worker เดียว
- ฟีเจอร์สินค้า/หมวดและรายการ popular เป็น batch (รีเฟรชด้วยการรัน `make features` และเทรนใหม่) ไม่มีระบบตรวจ drift
- ยังไม่มี screenshot Grafana ในรายงาน และ CI ยังไม่เคยรันบน GitHub

**งานต่อยอด:** ดึง candidate จาก session (ต้องเทรนใหม่ให้ Ranker เคยเห็น candidate แบบนี้), หลาย worker (ต้องทำ Prometheus แบบ multiprocess และให้ reload แจ้งทุก worker), timeout/load shedding, async Redis client, ANN แทน dot product เต็ม (เช่น Faiss), A/B test และ drift monitoring, feature store จริง (เช่น Feast), รายงานผลบน Test ครั้งเดียวกับโมเดลสุดท้าย

## 12. เอกสารเพิ่มเติม

| เอกสาร | เนื้อหา |
| --- | --- |
| [docs/progress.md](docs/progress.md) | บันทึกทุกขั้นเรียงตาม commit: ทำอะไร ตัดสินใจอะไร พบอะไร |
| [docs/runbook-training.md](docs/runbook-training.md) | วิธีรัน ETL, ฟีเจอร์, จูน และเทรนโมเดล (copy วางได้) |
| [docs/runbook-serving.md](docs/runbook-serving.md) | วิธีรัน Kafka + Redis + API และเปลี่ยน/ถอยโมเดล |
| [docs/runbook-loadtest.md](docs/runbook-loadtest.md) | วิธียิง k6 อ่าน Grafana และสร้างรายงาน |
| [docs/loadtest.md](docs/loadtest.md) | ผล load test ก่อน/หลังปรับปรุง พร้อมกราฟและข้อจำกัดของการวัด |
| [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) | EDA พร้อมคำอธิบายแต่ละหัวข้อ |
| [reports/](reports/) | ผลประเมินโมเดลและผลจูน ALS |

---

**Developed by Suphanat Chanlek**
