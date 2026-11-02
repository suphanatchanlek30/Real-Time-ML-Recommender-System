# Progress Log

บันทึกสิ่งที่ทำในแต่ละขั้น เรียงตาม commit เพื่อย้อนดูได้ว่าแต่ละอย่างมาจากไหน

| สถานะ | ขั้น | Commit |
| --- | --- | --- |
| ✅ | 0. ออกแบบระบบ (diagram) | `6871e71` |
| ✅ | 1. Setup repo | `d0d17f7` |
| ✅ | 2. EDA | `55e89e3`, `96ce9ac` |
| ✅ | 3. ETL และแบ่งข้อมูลตามเวลา | `4de5fa3` |
| ⬜ | 4. Evaluation และ baseline | - |
| ⬜ | 5. Feature engineering | - |
| ⬜ | 6. ALS + Ranker | - |
| ⬜ | 7. Serving, Streaming, Docker | - |
| ⬜ | 8. Load test และ Monitoring | - |

---

## 0. ออกแบบระบบ
- วาด diagram 3 ภาพ: Offline, Online, End-to-end ไว้ใน [diagrams/](diagrams/)
- เขียน README: เป้าหมาย, data, tech stack

## 1. Setup repo
- โครง `src/` (etl, features, training, evaluation, serving, streaming), `tests/`, `notebooks/`, `data/`
- `Makefile`, `pyproject.toml` (ruff + pytest), requirements, `.gitignore` (ไม่เก็บ `data/`)
- ดาวน์โหลด dataset ด้วย Kaggle API ลง `data/raw/`

## 2. EDA
ไฟล์: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb)

สิ่งที่พบและผลต่อการตัดสินใจ:

| พบ | ตัดสินใจ |
| --- | --- |
| 71% ของผู้ใช้มี event เดียว | ALS ใช้เฉพาะผู้ใช้ที่มี >= 2 events |
| ผู้ใช้ event สูงไม่ใช่บอทชัดเจน (ซื้อมากกว่าปกติ ~6 เท่า) | ไม่ตัดที่ p99.9 ตรงๆ ใช้กฎตามหัวข้อ 3 |
| ความยาว session ทำนายการซื้อได้ | ใช้เป็นฟีเจอร์ใน Ranker |
| ผู้ใช้ใน Val/Test เป็น warm แค่ ~15% | รายงานผล warm / cold แยกกัน |
| Top 1% ของสินค้า = 23% ของ events | popularity baseline จะแข็ง |

## 3. ETL และแบ่งข้อมูล
โค้ด: [src/etl/](../src/etl/) รันด้วย `make etl`

**ขั้นตอน** (ทำตามลำดับ):

1. อ่าน `events.csv` ด้วย schema ที่กำหนดเอง และตรวจด้วย Pandera
2. ลบแถวซ้ำ (460 แถว)
3. แปลง `timestamp` (epoch ms, UTC) เป็น datetime
4. ตัดผู้ใช้ผิดปกติ ถ้าเข้าข่ายข้อใดข้อหนึ่ง
   - event เกิน p99.99 (279)
   - event เกิน p99.9 (47) และไม่เคย addtocart/transaction เลย
   - ผลคือตัด 651 คน (159,845 events) ผู้ซื้อตัวจริงที่ event สูงแต่ไม่ถึงเพดานยังอยู่
5. ตรวจ schema ของข้อมูลสะอาด แล้วเขียน Parquet (zstd)
6. แบ่งตามเวลา
7. บันทึก `etl_report.json` (จำนวนที่ตัด, เกณฑ์, วันแบ่ง, SHA-256 ของไฟล์)

**การแบ่ง Train / Val / Test** (ห้ามสุ่ม เพราะจะรั่วข้อมูลอนาคตเข้า Train):

```text
|-------------- Train --------------|--- Val ---|--- Test ---|
3 พ.ค.                            4 ก.ย.      11 ก.ย.      18 ก.ย.
```

- Test = 7 วันสุดท้าย, Val = 7 วันก่อนหน้า, Train = ที่เหลือ (นับย้อนจาก event สุดท้าย)
- Val ใช้เลือก hyperparameter และตัดสินใจ promote, Test ดูครั้งเดียวตอนรายงานผลสุดท้าย

| ชุด | events |
| --- | --- |
| Train | 2,362,079 |
| Val | 125,044 |
| Test | 108,673 |

**ข้อจำกัดที่รู้อยู่:** การตัดบอทใช้จำนวน event ทั้งก้อน จึงใช้ข้อมูลอนาคตเล็กน้อย (ยอมรับได้เพราะเป็นแค่การคัดผู้ใช้ผิดปกติ ไม่ใช่การสร้างฟีเจอร์)

---

## ขั้นต่อไป
**4. Evaluation และ baseline:** เขียน Recall@K, NDCG@K, MAP@K และทำ popularity baseline เพื่อเป็นเส้นเทียบให้โมเดล
