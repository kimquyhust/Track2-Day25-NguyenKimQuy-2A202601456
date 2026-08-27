# NimbusAI — Báo cáo tối ưu chi phí GPU

**Kỳ báo cáo:** hàng tháng  
**Chi tiêu hiện tại (baseline):** $27,133  
**Chi tiêu sau tối ưu:** $15,325  
**Tiết kiệm dự kiến:** $11,808  (**44%**)  
**Đơn giá ($/1M token):** $6.488 → **$1.126**  (−83%)

## Tiết kiệm theo từng đòn bẩy

| Đòn bẩy | Tiết kiệm (USD) | Tỷ trọng |
|---|---:|---:|
| Inference: cascade + cache + batch | $1,212 | 10% |
| Mua hạ tầng: spot + reserved | $9,600 | 81% |
| Right-size GPU khai báo sai (util-lie) | $396 | 3% |
| Tắt GPU chạy không (idle) | $600 | 5% |

## Tiền đang rò rỉ ở đâu — và vì sao GPU-Util 98% vẫn là lãng phí

`gpu-h100-4`, `gpu-a10g-1` báo cáo GPU-Util từ 97% trở lên, nhìn qua thì như đang chạy hết công suất. Nhưng MFU chỉ 0.194. Hai con số này không mâu thuẫn — chúng đo hai thứ khác nhau. `nvidia-smi` báo GPU-Util là *tỷ lệ thời gian có ít nhất một kernel đang chạy*, nên một kernel bé xíu đang chờ dữ liệu từ HBM vẫn được tính là "bận". MFU mới đo phần FLOPs thực sự thu được so với FLOPs đỉnh mà bạn đã trả tiền.

Khoảng cách đó chính là tiền. Với `gpu-h100-4`, giá niêm yết là $2.50/giờ, nhưng vì chỉ khai thác được 19.4% năng lực tính toán nên chi phí thực cho mỗi *giờ FLOPs dùng được* là **$12.89** — đắt gấp 5.2 lần con số trên hoá đơn. Ba nguyên nhân quen thuộc, xếp theo tần suất gặp trong thực tế: batch size quá nhỏ khiến tensor core đói dữ liệu và GPU dành phần lớn thời gian chờ HBM (memory stall); chuỗi kernel ngắn khiến overhead khởi chạy kernel và đồng bộ chiếm tỷ trọng lớn; và dataloader trên CPU không theo kịp, GPU chạy rồi dừng theo nhịp I/O. Cả ba đều không thể phát hiện nếu chỉ nhìn GPU-Util — đó là lý do phải gắn MFU/MBU vào dashboard trước khi bàn chuyện mua thêm GPU.

Bên cạnh đó là lãng phí thô: `gpu-h100-5` (8h/ngày) vẫn bật nhưng gần như không làm gì, tương đương $20.00/ngày ($600/tháng) trả cho không khí. Đây là khoản cắt được ngay trong tuần này, không cần đổi kiến trúc.

## Right-sizing: mua băng thông, đừng mua FLOPs

Câu hỏi đúng không phải "GPU nào rẻ nhất theo $/giờ" mà là "GPU nào rẻ nhất trên đơn vị tài nguyên đang thực sự nghẽn". Với decode của LLM, thứ nghẽn là băng thông HBM và dung lượng KV-cache, không phải FLOPs. Bảng đơn giá cho thấy L4 rẻ nhất theo $/giờ ($0.80/giờ) nhưng lại là GPU **đắt nhất** theo $/TB-s — hạ cấp một workload memory-bound xuống L4 thì tiền thuê giảm nhưng throughput giảm nhiều hơn, và bạn phải thuê nhiều máy hơn để bù, tổng chi phí tăng.

| GPU | $/giờ | $/GB-VRAM | $/TB-s băng thông |
|---|---:|---:|---:|
| MI300X | $1.95 | $0.0102 | $0.368 |
| B200 | $5.09 | $0.0265 | $0.636 |
| H100 | $2.50 | $0.0312 | $0.746 |
| H200 | $3.95 | $0.0280 | $0.823 |
| A100 | $1.79 | $0.0224 | $0.895 |
| A10G | $1.00 | $0.0417 | $1.667 |
| L4 | $0.80 | $0.0333 | $2.667 |

Áp tiêu chí "MBU > MFU (memory-bound) và MFU < 30%", máy được đề xuất đổi:

| GPU | Hiện tại | Đề xuất | MBU/MFU | Tiết kiệm |
|---|---|---|---:|---:|
| `gpu-h100-4` | H100 ($2.50/h) | MI300X ($1.95/h) | 0.21/0.19 | −22% ($396/tháng) |

Bảng đòn bẩy chỉ ghi nhận $396/tháng từ các GPU đã bị gắn cờ util-lie — phần có bằng chứng rõ ràng nhất. Tổng cơ hội right-size trên toàn đội hình là $396/tháng; hai con số trùng nhau vì các GPU memory-bound còn lại đều bị chặn bởi ràng buộc VRAM. Lưu ý thực thi: đề xuất MI300X đúng về mặt kinh tế ($/GB-VRAM thấp nhất catalog, băng thông 5.3 TB/s) nhưng là hệ sinh thái ROCm — cần cộng thêm chi phí port kernel và một vòng benchmark trước khi chốt. Với A10G/L4, ràng buộc chặn lại là VRAM: không có hộp nào rẻ hơn mà vẫn chứa nổi KV-cache hiện tại.

## Kinh tế học của prompt caching: cần đọc lại bao nhiêu lần mới hoà vốn?

Prompt caching không miễn phí: ghi một entry tốn khoảng 1.25 lần một lượt đọc thường (bản 5 phút) hoặc 2.0 lần (bản 1 giờ), đổi lại mỗi lượt đọc sau đó chỉ còn 10% giá. Nên câu hỏi phải trả lời bằng số là: *một prompt cần được đọc lại bao nhiêu lần thì cache mới hoà vốn?*

| Chính sách TTL | Tier | Break-even (lượt đọc lại) | Thực đo | Kết luận | $ tiết kiệm/1M token input |
|---|---|---:|---:|---|---:|
| 5-min | small | 0.28 | 0.47 | Có lợi | $0.0343 |
| 5-min | large | 0.28 | 0.47 | Có lợi | $0.5119 |
| 1-hour | small | 1.11 | 0.47 | Lỗ | $-0.1157 |
| 1-hour | large | 1.11 | 0.47 | Lỗ | $-1.7381 |

Ngưỡng hoà vốn là 0,28 với cache 5 phút và 1,11 với cache 1 giờ. Đo trên `token_usage.csv` (số token cached chia cho số token input mới), traffic thực đang tái sử dụng 0.47 lượt/entry (gần như bằng nhau ở cả hai tier). Kết luận rất rõ: cache 5 phút có lãi (đang đóng góp $1.17/ngày, tương đương $35/tháng), còn cache 1 giờ thì lỗ — mức tái sử dụng hiện tại chưa tới một nửa ngưỡng cần thiết, bật lên là tự tăng hoá đơn.

Ngưỡng hoà vốn giống nhau ở cả hai tier vì phí ghi và chiết khấu đọc đều tính theo tỷ lệ trên giá input. Cái khác nhau là số tiền đặt cược: mỗi 1M token input được cache trên tier `large` tiết kiệm $0.5119, gấp 15 lần tier `small`. Vì vậy kỷ luật cache (system prompt ổn định, few-shot dùng chung, tách phần động xuống cuối prompt) nên áp trước hết cho tier đắt.

## Ngân sách reasoning: 8% traffic, 94% điện năng

Reasoning chỉ chiếm 8.4% số request nhưng ăn 16.5% chi phí và 94.0% năng lượng của tầng inference. Tỷ lệ lệch tới ~80 lần không phải vì model "nặng hơn", mà vì một request reasoning sinh ra một chuỗi suy luận nội bộ dài trước khi trả lời: output trung bình 3,875 token so với 641 token của traffic thường (6.0 lần). Mỗi token decode là một lượt đọc lại toàn bộ trọng số và KV-cache qua HBM — giai đoạn memory-bound, gần như không tái sử dụng được phép tính. Chuỗi dài gấp mấy lần thì số lượt đọc HBM, thời gian giữ GPU và điện năng cũng tăng theo bấy nhiêu, cộng dồn lại thành hệ số hàng chục lần.

| Kịch bản | Số request giữ reasoning | % traffic | Tiết kiệm $/ngày | Tiết kiệm Wh/ngày |
|---|---:|---:|---:|---:|
| Đặt trần 10% traffic | 201 | 8.4% | $0.00 | 0 |
| Đặt trần 5% traffic | 120 | 5.0% | $0.45 | 9,977 |
| Quy tắc routing: prompt ≥ 3,000 token | 60 | 2.5% | $0.63 | 19,411 |

Đọc bảng theo đúng thứ tự: đặt trần 10% traffic không tiết kiệm được gì ($0.00/ngày) đơn giản vì hiện tại đã ở 8.4% — trần đó không ràng buộc. Muốn có tác động thì phải siết chặt hơn. Quy tắc routing đề xuất: chỉ bật reasoning khi prompt đầu vào ≥ 3,000 token (proxy cho độ phức tạp: prompt dài thường là task nhiều ràng buộc, nhiều bước), phần còn lại rơi về chế độ thường. Quy tắc này giữ lại 60 request (2.5% traffic) và tiết kiệm $0.63/ngày ($19/tháng) cùng 19,411 Wh/ngày — tức 582 kWh/tháng. Cảnh báo đi kèm: đây là ước lượng theo độ dài output trung bình, nên trước khi bật phải chạy eval chất lượng trên đúng tập task bị hạ cấp; tiết kiệm năng lượng không đáng nếu tỷ lệ trả lời sai tăng lên.

## Chiến lược mua GPU: chính sách cũ vs. chính sách có tính rủi ro

| Job | GPU | Duty cycle | Tier chọn | Cam kết | On-demand | Sau tối ưu |
|---|---|---:|---|---|---:|---:|
| `job-train-llm` | H100 | 83% | spot | — | $12,000 | $7,704 |
| `job-train-embed` | A100 | 42% | spot | — | $2,148 | $1,393 |
| `job-finetune` | H100 | 25% | spot | — | $900 | $578 |
| `job-infer-chat` | A10G | 100% | reserved | 3 năm | $4,320 | $2,592 |
| `job-infer-rag` | A100 | 100% | reserved | 3 năm | $3,866 | $2,160 |
| `job-infer-search` | L4 | 75% | reserved | 1 năm | $1,728 | $1,296 |
| `job-dev-sandbox` | A10G | 33% | spot | — | $480 | $200 |
| `job-batch-eval` | H100 | 12% | spot | — | $225 | $144 |

Chính sách cũ (`recommend_tier`) chỉ nhìn hai biến: có gián đoạn được không, và duty cycle có vượt điểm hoà vốn không. Chính sách mới (`recommend_tier_v2`) định giá cả ba lựa chọn rồi chọn cái rẻ nhất, với hai bổ sung: (1) tỷ lệ gián đoạn spot theo từng loại GPU — H100 bị thu hồi ~8%/giờ trong khi L4 chỉ ~1.5%/giờ, nên chi phí rework sau mỗi lần mất máy được tính vào giá spot thay vì giả định chung 5%; (2) tách reserved 1 năm và 3 năm, chỉ cho phép cam kết 3 năm khi workload đã chứng minh duty ≥ 90% và chạy ≥ 30 ngày.

Kết quả: savings giảm từ 39.1% xuống 37.4% (chênh $440/tháng, tức 1.7 điểm phần trăm). Đây là chỗ dễ hiểu nhầm: chính sách mới *không tệ hơn*, nó chỉ ngừng ghi nhận khoản tiết kiệm mà chính sách cũ chưa trả giá. Cụ thể — `job-infer-search` chạy 75% thời gian nên bị hạ từ cam kết 3 năm xuống 1 năm. Cam kết 3 năm cho một job chạy 18 giờ/ngày là đổi 1.7 điểm phần trăm savings lấy rủi ro trả tiền cho 36 tháng một workload có thể biến mất sau hai quý; với H100 spot, giả định 5% thay vì 8% là giấu chi phí rework vào chỗ khuất. Số thấp hơn nhưng đáng tin hơn — và đó mới là số mang đi cam kết ngân sách.

## Phân bổ chi phí: đã sẵn sàng chargeback chưa?

| Team | $/ngày | $/tháng |
|---|---:|---:|
| assistant | $2.59 | $78 |
| search | $2.49 | $75 |
| eval | $1.79 | $54 |
| rag | $1.60 | $48 |

Tag coverage đang ở **92%**, vượt ngưỡng 80% nên cổng chargeback đã mở (`chargeback_ready = True`). Ngưỡng này không tuỳ tiện: dưới 80%, phần chi phí không gắn được tag phải phân bổ theo tỷ lệ ước lượng, và team bị tính tiền sẽ tranh cãi đúng — hoá đơn nội bộ mất uy tín ngay lần đầu sai. Khuyến nghị vận hành: chạy showback (chỉ hiển thị, chưa tính tiền) khoảng một chu kỳ để các team đối soát, đồng thời bật cảnh báo khi coverage tụt dưới 80%, rồi mới chuyển sang chargeback thật. File `outputs/focus_export.csv` đã xuất theo lược đồ FOCUS nên ghép được thẳng vào công cụ FinOps mà không cần ETL riêng.

## Tính bền vững

- Năng lượng mỗi truy vấn: 0.24 Wh
- Carbon mỗi truy vấn: 0.091 gCO2e
- Vùng sạch nhất: europe-north1
- Chuyển toàn bộ job có thể gián đoạn từ us-east-1 sang europe-north1 giảm 626.15 kgCO2e mỗi chu kỳ chạy (1,789 kWh), nhưng cộng thêm ~95 ms độ trễ.
- Vùng rẻ điện nhất là us-east-wa (tiết kiệm $116.28 tiền điện mỗi chu kỳ so với us-east-1).
- Lựa chọn cân bằng: us-east-wa — rẻ nhất trong nhóm vùng có cường độ carbon dưới trung vị, và chỉ xa hơn ~50 ms so với us-east-1.
- Reasoning chiếm 8.4% traffic nhưng 94.0% năng lượng truy vấn hằng ngày.

### So sánh các vùng triển khai

| Vùng | $/kWh | gCO2/kWh | Tiền điện | Carbon | Độ trễ |
|---|---:|---:|---:|---:|---:|
| us-east-1 | 0.120 | 380 | $214.68 | 679.82 kgCO2e | 15 ms |
| us-west-2 | 0.070 | 120 | $125.23 | 214.68 kgCO2e | 70 ms |
| europe-north1 | 0.090 | 30 | $161.01 | 53.67 kgCO2e | 110 ms |
| europe-central2 | 0.180 | 660 | $322.02 | 1,180.74 kgCO2e | 120 ms |
| us-east-wa | 0.055 | 90 | $98.39 | 161.01 kgCO2e | 65 ms |

## Kết luận và hành động theo thứ tự ưu tiên

1. **Tắt GPU idle — làm ngay tuần này.** $600/tháng, không rủi ro kỹ thuật, không cần đổi code. ROI cao nhất trên mỗi giờ công bỏ ra.
2. **Chốt lại danh mục mua hạ tầng.** $9,600/tháng (81% tổng tiết kiệm) — đòn bẩy lớn nhất. Spot cho job có checkpoint, reserved cho job chạy nền 24/7, và không cam kết 3 năm cho workload chưa đủ ổn định.
3. **Giữ cascade + batch + cache 5 phút cho inference.** $1,212/tháng, đưa đơn giá từ $6.488 xuống $1.126/1M token. Không bật cache 1 giờ cho tới khi tái sử dụng vượt 1.11 lượt/entry.
4. **Siết ngân sách reasoning theo quy tắc routing.** Thêm $19/tháng và 582 kWh/tháng, nhưng phải kèm eval chất lượng trước khi bật.
5. **Right-size `gpu-h100-4` (H100→MI300X) sau khi profile.** $396/tháng — để sau cùng vì cần một vòng benchmark (và port sang ROCm nếu chọn MI300X); MFU thấp đôi khi sửa được bằng tăng batch size mà không cần đổi phần cứng.
6. **Chạy showback trước, chargeback sau.** Coverage 92% đã đủ điều kiện, nhưng cần một chu kỳ đối soát và cảnh báo tự động khi coverage tụt dưới 80%.
7. **Lịch chạy theo carbon cho job gián đoạn.** europe-north1 nếu ưu tiên phát thải, us-east-wa nếu cần cân bằng tiền điện và độ trễ.

_Số liệu là ảnh chụp giá tháng 6/2026 — hãy dựng lại baseline trước khi áp dụng thực tế._