# Runbook: Load test และ Monitoring

ต้องมีระบบออนไลน์รันอยู่ก่อน (ดู [runbook-serving.md](runbook-serving.md)): `make up` และ `make replay`

## 1. เปิดระบบพร้อม Prometheus และ Grafana

```bash
make up                          # เปิด kafka, redis, api, consumer, prometheus, grafana
make replay                      # โหลดข้อมูลตั้งต้นและ replay event (~3 นาที)
```

| หน้า | ที่อยู่ (พอร์ตตั้งใน `.env`) | ดูอะไร |
| --- | --- | --- |
| Grafana | http://localhost:3000 (login `admin` / รหัสผ่านใน `GRAFANA_ADMIN_PASSWORD`, ไม่ login ก็ดูได้) | dashboard **Recommender: service overview** เป็นหน้าแรก |
| Prometheus | http://localhost:9090 | ลอง query เอง เช่น `rate(http_requests_total[30s])` และเช็ก targets ที่ Status → Targets |
| API metrics | http://localhost:8000/metrics | ค่าดิบที่ Prometheus เก็บ |

## 2. เตรียม user id ที่เหมือนทราฟฟิกจริง

```bash
make loadtest-users              # -> loadtest/users.json (6,000 id)
```

สัดส่วนที่สุ่ม: 15% ผู้ใช้ที่ ALS รู้จัก, 35% ผู้ใช้ที่มีประวัติใน Redis แต่ ALS ไม่รู้จัก, 50% ผู้ใช้ใหม่ (ตามสัดส่วนจริงที่ผู้ใช้ส่วนใหญ่เป็น cold) ต้องรันเมื่อโมเดลหรือข้อมูลเปลี่ยน

## 3. ยิง k6 (เปิด Grafana ดูไปพร้อมกัน)

```bash
make loadtest-smoke              # 1 VU, 30 วินาที: ระบบตอบถูกต้องไหม
make loadtest                    # ไต่ถึง 100 VUs แล้วค้างไว้ (~3.5 นาที) เกณฑ์: p95 < 100 ms, error < 1%
make loadtest-stress             # เพิ่มอัตรา request ไปเรื่อยๆ จนระบบเริ่มพัง (~3.5 นาที) เพื่อหาคอขวด
```

- **smoke / load** ใช้โมเดลแบบ *closed* (VU แต่ละตัวรอคำตอบก่อนยิงต่อ) พักคิด 0.1 วินาทีต่อรอบ
- **stress** ใช้โมเดลแบบ *open* (กำหนดอัตรา request ต่อวินาที) เพื่อให้เห็นอาการล้นจริง ถ้าระบบช้าลง request ใหม่ก็ยังเข้ามาเรื่อยๆ
- ผลสรุปพิมพ์ท้ายการรัน และไฟล์ดิบอยู่ที่ `results/<scenario>.json`
- ผ่าน/ไม่ผ่านดูจาก threshold ในบรรทัดสรุป (`PASS`/`FAIL`) ถ้าเกณฑ์ไม่ผ่าน k6 จะจบด้วย exit code ไม่ใช่ 0

## 4. อ่านผลใน Grafana

| Panel | ดูอะไร |
| --- | --- |
| Requests / s, Throughput by status | ปริมาณงานที่ตอบได้จริง และมีสถานะอื่นนอกจาก 200 ไหม |
| p95 / p99 latency, Latency percentiles | เส้นแดงคือเป้า 100 ms |
| Error rate | ต้องต่ำกว่า 1% |
| In flight | request ที่ค้างอยู่ ถ้าพุ่งขึ้นเรื่อยๆ แปลว่าระบบรับไม่ไหว |
| API CPU (cores) | ถ้าค้างที่ ~1.0 แปลว่า worker ตัวเดียวใช้ CPU เต็มแล้ว (CPU-bound) |
| Recommendations by source | ตรวจว่าทุกคำตอบมาจาก `ranker` ไม่ใช่ `popular_fallback` |
| Redis read time / Degraded answers | Redis ช้าหรือไม่ มีกี่ request ที่ต้องตอบโดยไม่มีฟีเจอร์ผู้ใช้ |
| Load generator (k6) | VUs และ latency ที่ฝั่ง k6 เห็น เทียบกับฝั่ง server |

ถ่าย screenshot ของ dashboard ระหว่างยิง (ช่วงเวลา "Last 15 minutes") เก็บไว้ประกอบรายงานได้

## 5. สร้างรายงานและกราฟจากผลที่ได้

```bash
# เก็บผลเป็นชุด: ผลดิบกับเวลาเริ่ม/จบของแต่ละรอบไว้ใน results/<ชื่อชุด>/
python -m src.serving.loadtest_report results/baseline --title "Before"
```

สคริปต์ดึงกราฟช่วงเวลานั้นจาก Prometheus (RPS, latency, error, CPU, in-flight) เซฟเป็น `docs/loadtest/<ชุด>-<scenario>.png` และสรุปตัวเลข รวมถึงหา "จุดที่ระบบเริ่มพัง" ของ stress test

## 6. ปิดระบบ

```bash
make down
```

## ถ้าติดปัญหา

| อาการ | วิธีแก้ |
| --- | --- |
| `loadtest/users.json` ไม่พบ | รัน `make loadtest-users` ก่อน |
| k6 ตอบ `connection refused` หรือ 503 | ระบบยังไม่พร้อม เช็ก `curl localhost:8000/health` และรัน `make replay` |
| Grafana ไม่มีข้อมูล | เช็ก http://localhost:9090/targets ว่า `api` และ `consumer` เป็น UP และปรับช่วงเวลาเป็น Last 15 minutes |
| ผล latency ไม่นิ่ง | ปิดโปรแกรมอื่นที่กิน CPU ระหว่างทดสอบ และรอให้ replay จบก่อนยิง |
| `make loadtest` ผ่านแต่ไม่เห็นเส้น k6 ใน Grafana | k6 ส่ง metrics ผ่าน Prometheus remote write: ต้องมี `--web.enable-remote-write-receiver` (ตั้งไว้ใน compose แล้ว) |
