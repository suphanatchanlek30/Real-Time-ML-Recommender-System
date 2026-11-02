# Progress Log

บันทึกสิ่งที่ทำในแต่ละขั้น เรียงตาม commit เพื่อย้อนดูได้ว่าแต่ละอย่างมาจากไหน

| สถานะ | ขั้น | Commit |
| --- | --- | --- |
| ✅ | 0. ออกแบบระบบ (diagram) | `6871e71` |
| ✅ | 1. Setup repo | `d0d17f7` |
| ✅ | 2. EDA | `55e89e3`, `96ce9ac` |
| ✅ | 3. ETL และแบ่งข้อมูลตามเวลา | `4de5fa3` |
| ✅ | 4. Evaluation และ baseline | `3386a8a` |
| ✅ | 5. Feature engineering | `b0f1e42` |
| ✅ | 6. ALS + Ranker | `c819ad9` |
| ✅ | 7. Serving, Streaming, Docker | `fa8ffa9`, `b306328`, `9da1f4b`, `c5b1c8b` |
| ✅ | 8. Load test และ Monitoring | `fc8667e`, `97fdfb4`, `a02f14d`, `37271b4`, `f0f4a9b` |

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

| ชุด | ช่วงเวลา (ปี 2015) |
| --- | --- |
| Train | 3 พ.ค. → 4 ก.ย. |
| Val | 4 ก.ย. → 11 ก.ย. |
| Test | 11 ก.ย. → 18 ก.ย. |

- Test = 7 วันสุดท้าย, Val = 7 วันก่อนหน้า, Train = ที่เหลือ (นับย้อนจาก event สุดท้าย)
- Val ใช้เลือก hyperparameter และตัดสินใจ promote, Test ดูครั้งเดียวตอนรายงานผลสุดท้าย

| ชุด | events |
| --- | --- |
| Train | 2,362,079 |
| Val | 125,044 |
| Test | 108,673 |

**ข้อจำกัดที่รู้อยู่:** การตัดบอทใช้จำนวน event ทั้งก้อน จึงใช้ข้อมูลอนาคตเล็กน้อย (ยอมรับได้เพราะเป็นแค่การคัดผู้ใช้ผิดปกติ ไม่ใช่การสร้างฟีเจอร์)

## 4. Evaluation และ baseline
โค้ด: [src/evaluation/](../src/evaluation/) รันด้วย `make eval-baseline` ผลอยู่ใน [reports/baseline_val.md](../reports/baseline_val.md)

**โจทย์ที่วัด:** ให้ผู้ใช้แต่ละคนได้รายการแนะนำ Top-K แล้วดูว่าตรงกับสินค้าที่เขาใส่ตะกร้า/ซื้อจริงในช่วง Val กี่ชิ้น (ground truth) ผู้ใช้ที่ไม่มี addtocart/transaction ในช่วงนั้นไม่ถูกนำมาวัด

**Metrics** (relevance แบบ 0/1, เฉลี่ยต่อผู้ใช้):

| Metric | ความหมาย |
| --- | --- |
| Recall@K | ถูกกี่ชิ้นจากทั้งหมดที่ผู้ใช้ทำจริง |
| Hit@K | มีถูกอย่างน้อย 1 ชิ้นไหม |
| NDCG@K | ถูกแล้วอยู่อันดับต้นหรือไม่ (ยิ่งอยู่บนยิ่งได้คะแนนมาก) |
| MAP@K | เฉลี่ย precision ณ ตำแหน่งที่ถูก |
| Coverage@K | สินค้ากี่ % ของแค็ตตาล็อกที่เคยถูกแนะนำ |

- ใช้ Recall@200 ประเมิน candidate generation ด้วย (ขั้น ALS) และ NDCG@10 เป็นตัวตัดสิน promote
- ทุกค่ามี **95% confidence interval** จาก bootstrap เพราะผู้ใช้ที่วัดได้มีแค่ ~1.9K คน ตัวเลขที่ช่วงทับกันถือว่าแยกกันไม่ได้
- รายงานแยก **warm** (เคยอยู่ใน Train) กับ **cold** (ไม่เคยเห็น)
- ทุก metric มี unit test เทียบกับค่าที่คำนวณมือ

**Baselines** (ไม่ personalise แนะนำรายการเดียวกันทุกคน):

| Baseline | วิธี |
| --- | --- |
| `random` | สุ่มสินค้า เป็นเส้นล่างสุดไว้เช็กว่าระบบวัดไม่ผิด |
| `popular_all` | สินค้ายอดนิยมจาก Train ทั้งหมด ถ่วงน้ำหนัก view 1 / addtocart 3 / transaction 5 |
| `popular_7d` | เหมือนกัน แต่นับเฉพาะ 7 วันสุดท้ายของ Train (เติมด้วย popular_all ถ้าไม่ครบ 200) |

**ผลบน Val** (ทุกผู้ใช้, n = 1,941: warm 289, cold 1,652):

| Model | Recall@10 | NDCG@10 | Recall@200 |
| --- | --- | --- | --- |
| random | 0.0005 | 0.0002 | 0.0018 |
| popular_all | 0.0106 | 0.0082 | 0.0753 |
| popular_7d | 0.0191 | 0.0102 | 0.1032 |

- popularity ชนะ random ชัดเจน ส่วน popular_7d ดีกว่า popular_all ที่ Recall@10 และ Recall@200 (ช่วงความเชื่อมั่นไม่ทับกัน) แต่ที่ NDCG@10 ยังแยกไม่ออก
- **เป้าหมายของโมเดล:** NDCG@10 ต้องดีกว่า 0.0102 และ Recall@200 ต้องดีกว่า 0.1032 อย่างมีนัยสำคัญ

**วินัยของ Test:** ยังไม่ได้รัน baseline บน Test (`--split test` เตรียมไว้แล้ว) จะรันครั้งเดียวตอนรายงานโมเดลสุดท้าย

**ข้อจำกัดที่รู้อยู่:** baseline ไม่ตัดสินค้าที่ผู้ใช้เคยทำไปแล้วออกจากรายการ และ ground truth มีแค่ ~1.9K ผู้ใช้ ผลจึงแกว่งง่าย

## 5. Feature engineering
โค้ด: [src/features/](../src/features/) รันด้วย `make features` (ต้อง `make etl` ก่อน) ผลอยู่ใน `data/features/`

**หลักการเดียว: point-in-time** ทุกฟังก์ชันรับ `as_of` และใช้เฉพาะ event ที่ `ts < as_of` ฟีเจอร์จึงไม่มีทางเห็นช่วงที่เรากำลังทำนาย (กัน leakage) มี test ที่ยืนยันว่าเติม event อนาคตเข้าไปแล้วผลไม่เปลี่ยน (ลองแก้โค้ดให้รั่วแล้ว test ล้มจริง)

**Snapshot** (สร้างตารางฟีเจอร์ 3 ชุด ขอบเขตดึงจาก `split_boundaries` ตัวเดียวกับ ETL):

| Snapshot | as_of | labels | ใช้ทำ |
| --- | --- | --- | --- |
| `rank_1` ... `rank_4` | val_start - 7k วัน | 7 วันถัดไป (อยู่ใน Train ทั้งหมด) | เทรน Ranker (ฟีเจอร์จากอดีต, label จาก "อนาคต") ใช้ 4 ช่วงต่อเนื่องเพื่อให้มีผู้ใช้ที่มี label มากพอ |
| `val` | val_start | ช่วง Val | เลือกโมเดล |
| `test` | test_start (history = Train + Val) | ช่วง Test | รายงานผลสุดท้าย |

**ฟีเจอร์ 38 ตัว** (นิยามทั้งหมดใน [schema.py](../src/features/schema.py) ที่เดียว ใช้สร้าง Pandera schema และ `schema.json` อัตโนมัติ):

| กลุ่ม | ตัวอย่าง | หมายเหตุ |
| --- | --- | --- |
| user (11) | จำนวน view/cart/buy, sessions, recency, decayed interest | เก็บเป็น *state* ที่อัปเดตทีละ event ได้ แล้วค่อย `derive` ตอนใช้งาน |
| item (16) | popularity 1/7/28 วัน, trend, unique users, อายุ, conversion แบบ smoothed, category, available | batch อย่างเดียว |
| category (3) | popularity ของหมวด, จำนวนสินค้า | รวมจาก item features |
| pair (5) | user เคย view/cart/buy สินค้านี้กี่ครั้ง, ห่างกี่วัน, สัดส่วนหมวดที่ผู้ใช้สนใจ | คำนวณตอนให้คะแนน candidate |
| candidate (3) | คะแนน ALS, อันดับที่ดึงมา, ดึงจาก ALS หรือ popular fallback | เพิ่มตอนทำขั้น 6 |

**ตัดสินใจสำคัญ:**
- **offline = online:** user features แยกเป็น state (counts, เวลาแรก/ล่าสุด, decayed sums, 20 item ล่าสุด) กับ `derive` (recency, ratio) ไม่เก็บค่าอย่าง "days since last" เพราะเก่าทันทีที่เขียน `update_user_state` (ทีละ event สำหรับ consumer) ให้ผลเท่ากับ batch ตาม test
- **Decayed counts** (`exp(-อายุ/7 วัน)`) แทนหน้าต่างคงที่ เพราะอัปเดตทีละ event ได้
- **Conversion แบบ smoothed:** 66% ของสินค้ามี event ไม่เกิน 5 ครั้ง อัตรา raw จะหลอก จึงดึงเข้าหาค่าเฉลี่ยรวม
- **category / available แบบ as-of:** ดึงจากไฟล์ item_properties ครั้งเดียวใน ETL (4.2 MB) แล้วใช้ค่าล่าสุดก่อน `as_of` ไม่ใช้ "ค่าล่าสุดทั้งไฟล์" ที่รั่วอนาคต
- ใช้ `EVENT_WEIGHTS` จาก [src/config.py](../src/config.py) ที่เดียว ร่วมกับ baseline

**ข้อจำกัดที่รู้อยู่:**
- ผู้ใช้ cold ไม่มีแถว user features (ค่าเป็น null ให้ LightGBM จัดการ) และ ~21% ของสินค้าไม่รู้หมวด ณ เวลานั้น (ใช้ -1)
- **Session features วัดผลไม่ได้ในโปรโตคอลนี้:** เราทำนาย 7 วันข้างหน้าจาก snapshot ต้นช่วง "session ปัจจุบัน" จึงเก่าหรือว่าง ฟีเจอร์ `recent_items` และ state ทำไว้เพื่อให้ตรงกับ Redis แต่ไม่ควรคาดหวังว่าจะเห็นผลบน Val
- Pair features ช่วยได้เฉพาะถ้า candidate set ไม่ตัดสินค้าที่ผู้ใช้เคยดูออก (ต้องระวังตอนทำ ALS + Ranker)

## 6. ALS + Ranker
โค้ด: [src/training/](../src/training/) วิธีรันอยู่ใน [runbook-training.md](runbook-training.md) (`make tune-als`, `make train`, `make mlflow-ui`)

**ไหลของการเทรน:**

1. **ALS** (`implicit`) เทรนจาก interaction ที่ถ่วงน้ำหนัก (view 1 / addtocart 3 / transaction 5) ของผู้ใช้/สินค้าที่มี event >= 2 ครั้ง ให้ score ด้วย dot product ของ factors (serving ใช้แค่ numpy ไม่ต้องมี `implicit`)
2. **Candidate 200 ตัวต่อผู้ใช้:** ผู้ใช้ที่ ALS รู้จักได้จาก ALS (ไม่ตัดสินค้าที่เคยดูออก) ที่เหลือ (cold) ได้ popular 7 วันล่าสุดเป็น fallback
3. **Ranker:** LightGBM LambdaRank เรียง 200 ตัวนั้นใหม่ด้วยฟีเจอร์ 38 ตัว เทรนบน 4 snapshot ต่อเนื่อง (`rank_4`..`rank_1`) โดย early stopping บน snapshot ใหม่สุด แล้ว refit ทั้งหมด
4. **Evaluation gate:** บน Val โมเดลใหม่ต้องมี NDCG@10 ดีกว่า champion เดิม (หรือ popular_7d ถ้ายังไม่มี) และ Recall@200 ไม่แย่ลง
5. **MLflow:** log params, metrics, tags (git commit, hash ข้อมูล) และ bundle (`als.npz`, `ranker.txt`, `schema.json`, `popular.json`) ถ้าผ่าน gate ลงทะเบียนเวอร์ชันและย้าย alias `champion` (ใช้ SQLite เป็น backend จึงมี Registry โดยไม่ต้องมี server)

**จูน ALS** (`reports/als_tuning.md`): ลอง 12 ชุด ได้ factors 128 / regularization 0.01 / alpha 40 (Recall@200 0.37 เทียบกับ 0.33 ของ factors 64 บนผู้ใช้ที่ ALS รู้จัก)

**ผลบน Val** (n = 1,941: warm 289, cold 1,652 ดูตารางเต็มที่ [reports/train_val.md](../reports/train_val.md)):

| Model | NDCG@10 | Recall@10 | Recall@200 |
| --- | --- | --- | --- |
| popular_7d | 0.0102 | 0.0191 | 0.1032 |
| ALS | 0.0170 | 0.0295 | 0.1263 |
| **ALS + Ranker** | **0.0333** | **0.0438** | 0.1263 |

| กลุ่ม | NDCG@10: popular_7d → ALS → ALS + Ranker |
| --- | --- |
| warm (289) | 0.0090 → 0.0550 → **0.1318** |
| cold (1,652) | 0.0104 → 0.0104 → **0.0160** |

- **Gate: ผ่าน** ลงทะเบียน `recsys-bundle` v1 และตั้ง alias `champion`
- NDCG@10 รวมสูงกว่า baseline ~3.3 เท่า และช่วงความเชื่อมั่นไม่ทับกับ baseline
- Recall@200 ของ ALS และ ALS + Ranker เท่ากัน เพราะ Ranker แค่เรียง 200 ตัวเดิมใหม่

**ข้อสังเกตและข้อจำกัด:**
- **ผู้ใช้ส่วนใหญ่เป็น cold** ALS รู้จักผู้ใช้แค่ ~10-15% ของผู้ใช้ที่วัด จึงไม่ช่วยกลุ่ม cold เลย (ผลเท่า baseline) ส่วนที่ดีขึ้นของกลุ่ม cold (0.0104 → 0.0160) มาจาก Ranker ที่เรียง popular ใหม่ด้วยฟีเจอร์สินค้า และช่วงความเชื่อมั่นยังทับกับ baseline เล็กน้อย
- **กลุ่ม warm ดีขึ้นมาก** ส่วนใหญ่มาจาก pair features (ผู้ใช้เคยดู/ใส่ตะกร้าสินค้านั้นมาก่อน) ซึ่งสะท้อนพฤติกรรมกลับมาซื้อซ้ำ ไม่ใช่การค้นพบสินค้าใหม่
- **ข้อมูลเทรน Ranker น้อย:** ~1,100 ผู้ใช้ที่มี label ตรงกับสินค้าใน candidate (จาก 4 snapshot) ตัวเลข `ranker_holdout_ndcg10` ใน MLflow (~0.31) วัดเฉพาะผู้ใช้ที่มี label จึงสูงกว่าผลจริงบน Val และเทียบกันไม่ได้
- **จูน ALS วัดจากผู้ใช้ 182 คน** และค่าที่ดีที่สุด (factors 128) อยู่ขอบของตาราง ยังลอง 256 ได้
- Val ถูกใช้ทั้งจูนและตัดสิน gate จึงมีความเอนเอียงเล็กน้อย ตัวเลขที่เชื่อได้จริงคือ Test ซึ่ง **ยังไม่ได้แตะ**

## 7. Serving, Streaming, Docker
โค้ด: [src/serving/](../src/serving/), [src/streaming/](../src/streaming/), [docker-compose.yml](../docker-compose.yml) วิธีรันอยู่ใน [runbook-serving.md](runbook-serving.md) (`make up`, `make replay`, `make smoke`)

**ระบบที่ได้:** `Kafka (user-events) → consumer → Redis → FastAPI` โดย API โหลดโมเดลจาก `models/champion/` (ดึงจาก MLflow ด้วย `make pull-model`)

| ส่วน | ทำอะไร |
| --- | --- |
| `recommender.py` | ดึง candidate 200 ตัว (ALS สำหรับผู้ใช้ที่ ALS รู้จัก, popular สำหรับที่เหลือ) → สร้างฟีเจอร์ 38 ตัว → Ranker เรียง |
| `feature_store.py` | โครง Redis, ตัวจัดการ event ทีละตัว, ตัวโหลด state ตั้งต้น, ตัวอ่านผู้ใช้ (1 round trip) |
| `app.py` | `/recommend`, `/health`, `/metrics` (Prometheus), `/admin/reload` |
| `streaming/producer.py` | replay event เรียงตามเวลา key = `visitor_id` |
| `streaming/consumer.py` | ตรวจ schema แล้วอัปเดต Redis ทีละ batch |

**การตัดสินใจสำคัญ (และต่างจากแผนเดิมตรงไหน):**

1. **Event-time clock ไม่ใช่เวลาจริง** ข้อมูลเป็นปี 2015 ถ้าใช้เวลาเครื่อง recency จะเป็น ~4,000 วันและ decayed sums เป็น 0 ทุกคน จึงเก็บเวลาของ event ล่าสุดไว้ใน Redis (`clock`) และคำนวณฟีเจอร์ ณ เวลานั้น
2. **จุดเริ่มต้นเดียวกับ champion:** โหลด state ตั้งต้น ณ `val_start` (ตรงกับที่ ALS และ Ranker เทรน) แล้ว replay เฉพาะ event หลังจากนั้น ไม่ต้องสร้าง snapshot ใหม่ และ `make replay` ล้าง Redis ก่อนทุกครั้งจึงไม่นับซ้ำ
3. **ไม่ใช้ TTL** (แผนเดิมมี) เพราะ state ที่หมดอายุจะทำให้ฟีเจอร์ online ต่างจากที่โมเดลเรียนมา
4. **กัน skew ของ pair features:** แผนเดิมเก็บแค่ 20 สินค้าล่าสุด แต่ตอนเทรนนับจากประวัติทั้งหมด จึงเก็บ hash จำนวน view/cart/buy ต่อสินค้า (`user:{id}:items`) และจำนวนต่อหมวด (`user:{id}:cats`) เพิ่ม (`user:{id}:recent` ยังเก็บไว้ตามข้อตกลง)
5. **ฟีเจอร์ผู้ใช้ใช้โค้ดชุดเดียวกับ offline** (`derive_user_features`) ส่วนฟีเจอร์สินค้าโหลดเป็น numpy ตอน startup เพื่อให้ request ละไม่กี่ ms
6. **Redis ช้า/ล่ม:** timeout 50 ms แล้วตอบด้วย Ranker โดยไม่มีฟีเจอร์ผู้ใช้ (`degraded: true`); Ranker พัง → ตอบ popular list (`source: popular_fallback`)
7. **ตั้งค่าด้วย `.env`:** พอร์ต, image, topic, timeout, snapshot, token ฯลฯ อยู่ใน `.env` ที่เดียว (`.env.example` เป็นแม่แบบที่ commit, `.env` ไม่ถูก commit) Dockerfile ไม่มีค่าตั้งค่าอยู่เลย compose ส่งค่าเข้า container ผ่าน `${...}` และโค้ดบนเครื่องอ่าน `.env` เอง (`src/env.py`) ทดสอบจริงโดยเปลี่ยนทุกค่าจากค่าเริ่มต้น (พอร์ต 8001/6380/9095, topic, partition, อัตรา replay, token) แล้วทั้งระบบยังทำงานและผลถูกต้อง มี test กันไม่ให้ compose อ้างตัวแปรที่ไม่อยู่ใน `.env.example`
8. **Reload ปลอดภัย:** สร้างโมเดลใหม่ให้เสร็จและตรวจก่อน (ลำดับฟีเจอร์ต้องตรง `spec.py`, snapshot ต้องตรง) ถ้าไม่ผ่านตอบ 409 และใช้โมเดลเดิมต่อ ทดสอบกับ stack จริงแล้ว

**ผลที่วัดได้** (MacBook, Docker Desktop):

| รายการ | ค่า |
| --- | --- |
| โหลด state ตั้งต้นเข้า Redis | 1,279,697 ผู้ใช้ใน 28 วินาที, 5.1 ล้าน keys, ใช้ RAM ~720 MB |
| Replay | 233,717 events ที่ 2,000/วินาที (~2 นาที) consumer ตามทัน (lag 0) |
| Latency ของ `/recommend` ผ่าน HTTP | p50 3.9 ms, p95 9.7 ms, p99 14.1 ms (ยิงทีละ request 612 ครั้ง ยังไม่ใช่การทดสอบภายใต้โหลด) |
| ส่วนที่กินเวลาในโค้ด | ดึง candidate ~1.7 ms, สร้างฟีเจอร์ ~0.7 ms, Ranker ~0.1 ms |

**การทดสอบ:**
- **offline = online:** บนข้อมูลสังเคราะห์ ฟีเจอร์ผู้ใช้และ pair features ที่คำนวณจาก Redis (เติมด้วย stream) เท่ากับ offline; บนโมเดล champion จริงกับผู้ใช้ 120 คน ฟีเจอร์และลำดับที่ API แนะนำเท่ากับเส้นทางตอนเทรนทุกตัว
- API: contract, validation, Redis ล่ม, Ranker พัง, reload, token
- `make smoke`: ยิง event เข้า Kafka จริงแล้วเช็กว่าไปถึง Redis และ API ตอบได้

**บั๊กที่เจอจากการทดสอบจริง:** ตอนจำลอง consumer ล้ม (docker kill) กลาง replay Kafka ส่ง batch เดิมซ้ำ ทำให้ผู้ใช้ **343 คน** ถูกนับ event ซ้ำ (ตรวจเทียบทั้ง 1.4 ล้านคนกับข้อมูลจริง) แก้โดยเก็บ offset ล่าสุดที่ใช้แล้วไว้ใน Redis ใน transaction เดียวกับการอัปเดต state หลังแก้ ทดสอบซ้ำแบบเดียวกันแล้วผิด 0 คน มี unit test ครอบคลุม

**ข้อจำกัดที่รู้อยู่:**
- **ผลแนะนำเปลี่ยนตาม session แต่เปลี่ยนไม่มาก** ผู้ใช้ cold ได้ candidate จาก popular เท่านั้น สินค้าที่เพิ่งดู/ใส่ตะกร้าจึงไม่ถูกดึงเข้ามา ฟีเจอร์ที่เปลี่ยนมีผลแค่ผ่านฟีเจอร์ผู้ใช้ ส่วนผู้ใช้ที่ ALS รู้จักจะเห็น pair features ทำงานเต็มที่ การดึง candidate จาก session (ต้องเทรนใหม่เพื่อให้ Ranker เคยเห็น candidate แบบนี้) เป็นงานต่อยอด
- โปรโตคอลประเมินแบบ snapshot ต้นสัปดาห์ จึงวัดผลของ session features ไม่ได้ (เขียนไว้ตั้งแต่ขั้น 5)
- Popular list ใน bundle ถูกตรึง ณ เวลาเทรน; ฟีเจอร์สินค้า/หมวดเป็น batch (รีเฟรชด้วยการรัน `make features` ใหม่)
- Replay วิ่งเลย `val_start` ไปถึงช่วง Test (4-18 ก.ย.) เพราะแค่ให้ stack มีข้อมูลสด ไม่ได้ใช้ประเมินอะไร
- MLflow ยังอยู่บนเครื่อง (SQLite) ไม่อยู่ใน compose; API worker เดียว; `/admin/reload` ป้องกันด้วย `ADMIN_TOKEN` ถ้าตั้งไว้เท่านั้น
- Kafka/consumer ทดสอบกับ stack จริงบนเครื่องเดียว ไม่ได้ทดสอบหลาย consumer หรือหลาย broker

## 8. Load test และ Monitoring
โค้ด: [monitoring/](../monitoring/), [loadtest/k6.js](../loadtest/k6.js), [.github/workflows/ci.yml](../.github/workflows/ci.yml) วิธีรันอยู่ใน [runbook-loadtest.md](runbook-loadtest.md) และผลเต็มอยู่ใน [loadtest.md](loadtest.md)

| ส่วน | ทำอะไร |
| --- | --- |
| Prometheus | scrape API และ consumer ทุก 5 วินาที และรับ metrics ที่ k6 ส่งมา (remote write) |
| Grafana | dashboard `Recommender: service overview` ถูกสร้างอัตโนมัติจากไฟล์: RPS, error rate, p50/p95/p99 (เส้นเป้า 100 ms), แหล่งที่มาของคำตอบ, เวลาอ่าน Redis, CPU/RAM, Kafka lag, และฝั่ง k6 |
| k6 | smoke / load / stress พร้อม threshold (p95 < 100 ms, error < 1%) ผู้ใช้ที่สุ่มมาจาก 3 กลุ่มตามสัดส่วนจริง |
| รายงาน | สคริปต์ดึงกราฟจาก Prometheus ของแต่ละรอบ และหาจุดที่ระบบเริ่มพัง |
| CI | GitHub Actions: ruff, pytest, ตรวจ compose, build image ทั้งสอง |

**ผลที่ได้ (ก่อน → หลังปรับปรุง):**

| | ก่อน | หลัง |
| --- | --- | --- |
| Load 100 VUs: p95 | 166 ms (ไม่ผ่าน) | **30 ms** (ผ่าน) |
| request / วินาที | 401 | **695** |
| Stress: อัตราที่รักษา p95 < 100 ms | ~300 / วินาที | **~1,360 / วินาที** |
| CPU ของ API | 9.6 คอร์ | 1.4 คอร์ |

**ปรับปรุงอะไรไปบ้าง (ตามหลักฐานที่วัดได้):**
1. ผลก่อนปรับเจอว่า API ใช้ CPU ~9.4 คอร์เพื่อรับ ~500 request/วินาที เพราะ thread ของ numpy/LightGBM แย่งกัน จำกัดเป็น 1 thread (`COMPUTE_THREADS` ใน `.env`) CPU ลด 5 เท่า แต่ throughput แทบไม่ขึ้น (401 → 427)
2. Profile ต่อ request ชี้ว่าการสร้าง DataFrame polars เพื่อคำนวณฟีเจอร์ผู้ใช้กิน ~0.94 ms จึงเปลี่ยนเป็น Python ล้วน (test เทียบกับโค้ด offline) พร้อมปรับการหา pair features, middleware, และคำนวณคำตอบของผู้ใช้ใหม่ครั้งเดียวต่อโมเดล ผลคือ 401 → 695 request/วินาที และ p95 166 → 30 ms

**ข้อจำกัดที่รู้อยู่:**
- ทุกอย่างรันบนเครื่องเดียว (k6 อยู่เครื่องเดียวกับ server) และวัดแต่ละชุดแค่รอบเดียว ตัวเลขเหมาะกับการเทียบก่อน/หลัง ไม่ใช่ความจุจริง
- ระบบล้นแล้ว "ช้าลง" ไม่ได้ "ตอบ error" (ไม่มี load shedding) และยังมี worker เดียว: ความจุเพิ่มได้อีกด้วยหลาย worker แต่ต้องทำ Prometheus แบบ multiprocess และให้ reload แจ้งทุก worker
- ยังไม่มี screenshot ของ Grafana ในรายงาน (ตรวจแล้วว่า query ของทุก panel ทำงานและ dashboard โหลดได้ แต่ยังไม่ได้ดูภาพที่วาดจริง) ใช้กราฟที่สร้างจาก Prometheus ใน `docs/loadtest/` แทน
- GitHub Actions ยังไม่เคยรันบน GitHub (จำลองบนเครื่องใน venv สะอาดไม่มีข้อมูลแล้วผ่าน) จะรันครั้งแรกเมื่อเปิด PR

---

## ขั้นต่อไป
ครบทุกขั้นตามแผนแล้ว งานต่อยอดที่เหลืออยู่ในหัวข้อ ข้อจำกัด ของแต่ละขั้น เช่น หลาย worker, session-based retrieval, รันบน Test set ครั้งเดียวสำหรับโมเดลสุดท้าย
