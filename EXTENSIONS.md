# Phần mở rộng "Your Turn" — Lab 25

Cả **5/5** extension đã được làm, mỗi phần đều chạy được, có số đo trước/sau và
có unit test riêng trong [`tests/test_extensions.py`](tests/test_extensions.py)
(13 test, không sửa 15 test gốc).

Chạy lại toàn bộ: `python verify.py` (11/11) · `pytest -q` (28 passed) ·
`python missions/run_all.py`.

---

## 1. Chính sách mua GPU có tính rủi ro — `pricing.recommend_tier_v2()`

**File:** [`finops/pricing.py`](finops/pricing.py) · dùng trong [`missions/m3_purchasing.py`](missions/m3_purchasing.py)

Chính sách cũ chỉ xét *interruptible* và *duty cycle*. Chính sách mới định giá cả
ba lựa chọn rồi chọn cái rẻ nhất, với hai yếu tố bổ sung:

- **Tỷ lệ gián đoạn spot theo từng loại GPU** (`SPOT_INTERRUPT_RATE`): H100 ~8%/giờ,
  A100 5%, A10G 2%, L4 1.5% — chi phí rework được tính vào giá spot thay vì dùng
  chung một con số 5%.
- **Tách reserved 1 năm vs 3 năm**: chỉ cho cam kết 3 năm khi duty ≥ 90% **và**
  workload chạy ≥ 30 ngày.

| | Chính sách cũ | Chính sách mới |
|---|---:|---:|
| Chi phí tối ưu / tháng | $15,627 | $16,067 |
| Savings so với on-demand | **39.1%** | **37.4%** |

**Savings thay đổi thế nào và tại sao?** Giảm 1.7 điểm phần trăm (+$440/tháng).
Toàn bộ chênh lệch đến từ `job-infer-search` (L4, duty 75%): chính sách cũ ghi nhận
giá reserved 3 năm cho một workload chưa đủ ổn định để cam kết 36 tháng, còn chính
sách mới hạ xuống 1 năm. Số mới thấp hơn nhưng là số dám mang đi cam kết ngân sách.

## 2. Right-sizing theo MBU — `$/GB-VRAM` và `$/TB-s`

**File:** [`finops/metrics.py`](finops/metrics.py) (`dollars_per_gb_vram`, `dollars_per_tbs`,
`is_memory_bound`, `rightsize_candidate`) · dùng trong [`missions/m1_efficiency_audit.py`](missions/m1_efficiency_audit.py)

Tiêu chí chọn máy thay thế: rẻ hơn theo $/giờ **và** vẫn đủ băng thông + VRAM với
20% headroom so với mức đo được trong telemetry.

| GPU | $/giờ | $/GB-VRAM | $/TB-s |
|---|---:|---:|---:|
| MI300X | $1.95 | $0.0102 | **$0.368** |
| H100 | $2.50 | $0.0312 | $0.746 |
| A100 | $1.79 | $0.0224 | $0.895 |
| L4 | $0.80 | $0.0333 | **$2.667** |

Kết quả: `gpu-h100-4` (MBU 0.21 > MFU 0.19) → **MI300X**, −22% tức **$396/tháng**.
`gpu-a10g-1` bị chặn bởi VRAM: không có hộp nào rẻ hơn còn chứa nổi KV-cache.

**Tại sao không chọn GPU rẻ nhất theo `$/GPU-hr`?** Vì L4 rẻ nhất theo giờ ($0.80)
nhưng đắt nhất theo băng thông ($2.667/TB-s) — với decode memory-bound, hạ xuống L4
làm throughput rơi nhanh hơn giá thuê, phải thuê nhiều máy hơn và tổng chi phí tăng.

## 3. Kinh tế học cache — `cache_is_worth_it()` + break-even đo từ dữ liệu thật

**File:** [`finops/pricing.py`](finops/pricing.py) (`cache_is_worth_it`,
`cache_break_even_reads`, `cache_savings_per_million`, `CACHE_TTL_POLICIES`) ·
áp dụng trong [`missions/m2_inference_levers.py`](missions/m2_inference_levers.py)

Cache chỉ được bật cho tier nào có mức tái sử dụng **đo được** vượt ngưỡng hoà vốn
của tier đó (`observed_cache_reads()` = token cached / token input mới).

| TTL | Tier | Break-even | Thực đo | Kết luận | $/1M token input |
|---|---|---:|---:|---|---:|
| 5 phút | small | 0.28 | 0.47 | Có lợi | +$0.0343 |
| 5 phút | large | 0.28 | 0.47 | Có lợi | +$0.5119 |
| 1 giờ | small | 1.11 | 0.47 | **Lỗ** | −$0.1157 |
| 1 giờ | large | 1.11 | 0.47 | **Lỗ** | −$1.7381 |

**Cần đọc lại bao nhiêu lần? Dataset có đạt ngưỡng không?** Cache 5 phút hoà vốn ở
**0.28 lượt đọc lại**, cache 1 giờ ở **1.11 lượt**. Traffic thực chỉ tái sử dụng
**0.47 lượt/entry** → bật cache 5 phút (đang đóng góp $1.17/ngày ≈ $35/tháng), **không**
bật cache 1 giờ. Ngưỡng giống nhau ở hai tier vì phí ghi/chiết khấu đều theo tỷ lệ;
thứ khác nhau là số tiền đặt cược — tier `large` gấp 15 lần tier `small`.

## 4. Ngân sách reasoning + quy tắc routing

**File:** [`missions/m2_inference_levers.py`](missions/m2_inference_levers.py)
(`reasoning_scenarios`) · đưa vào báo cáo ở [`missions/m5_report.py`](missions/m5_report.py)

Hiện trạng: **201/2400 request (8.4% traffic)** → **16.5% chi phí** và **94.0% năng lượng**
của tầng inference.

| Kịch bản | Giữ reasoning | % traffic | Tiết kiệm $/ngày | Tiết kiệm Wh/ngày |
|---|---:|---:|---:|---:|
| Đặt trần 10% traffic | 201 | 8.4% | $0.00 | 0 |
| Đặt trần 5% traffic | 120 | 5.0% | $0.45 | 9,977 |
| **Quy tắc: prompt ≥ 3,000 token** | 60 | 2.5% | **$0.63** | **19,411** |

**Reasoning chiếm bao nhiêu % và tại sao tốn ~80× năng lượng?** 8.4% traffic. Trần 10%
không tiết kiệm được gì vì hiện đã ở dưới ngưỡng — phải siết bằng quy tắc routing thì
mới có tác động ($19/tháng + 582 kWh/tháng). Tốn ~80× vì mỗi request reasoning sinh
chuỗi suy luận nội bộ dài (output trung bình 3,875 token so với 641 token của traffic
thường); mỗi token decode phải đọc lại toàn bộ trọng số + KV-cache qua HBM — pha
memory-bound gần như không tái sử dụng được phép tính, nên độ dài chuỗi nhân thẳng vào
số lượt đọc HBM, thời gian giữ GPU và điện năng.

## 5. Lịch chạy nhận thức carbon

**File:** [`finops/sustainability.py`](finops/sustainability.py) (`region_comparison`,
`region_scorecard`, `REGION_LATENCY_MS`) · dùng trong [`missions/m3_purchasing.py`](missions/m3_purchasing.py)

Toàn bộ job `interruptible=1` tiêu thụ **1,789 kWh** mỗi chu kỳ chạy.

| Vùng | $/kWh | gCO2/kWh | Tiền điện | Carbon | Độ trễ |
|---|---:|---:|---:|---:|---:|
| us-east-1 (hiện tại) | 0.120 | 380 | $214.68 | 679.82 kg | 15 ms |
| us-west-2 | 0.070 | 120 | $125.23 | 214.68 kg | 70 ms |
| europe-north1 | 0.090 | 30 | $161.01 | **53.67 kg** | 110 ms |
| europe-central2 | 0.180 | 660 | $322.02 | 1,180.74 kg | 120 ms |
| us-east-wa | **0.055** | 90 | **$98.39** | 161.01 kg | 65 ms |

**Vùng nào "tối ưu" thực sự?** Tuỳ tiêu chí công ty ưu tiên:
sạch nhất → `europe-north1` (−626 kgCO2e/chu kỳ, nhưng +95 ms độ trễ và ràng buộc
data residency EU); rẻ điện nhất → `us-east-wa` (−$116/chu kỳ); cân bằng →
`us-east-wa`, vì nó rẻ nhất trong nhóm vùng có cường độ carbon dưới trung vị và chỉ
xa hơn 50 ms. `europe-central2` là bẫy: điện đắt nhất **và** lưới bẩn nhất.
