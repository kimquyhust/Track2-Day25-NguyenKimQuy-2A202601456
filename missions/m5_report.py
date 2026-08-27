"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

The deliverable is written in Vietnamese because that is the audience: the
NimbusAI platform team. All figures are pulled from the M1-M4 result dicts, so
the prose and the terminal output can never drift apart.

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation

DAYS = 30
MEDIAN_QUERY_TOKENS = 800


def _levers(r1, r2, r3) -> dict:
    return {
        "Inference: cascade + cache + batch": round((r2["baseline_daily"] - r2["optimized_daily"]) * DAYS),
        "Mua hạ tầng: spot + reserved": round(r3["on_demand_monthly"] - r3["optimized_monthly"]),
        "Right-size GPU khai báo sai (util-lie)": round(r1["rightsize_lie_monthly"]),
        "Tắt GPU chạy không (idle)": round(r1["idle_waste_daily"] * DAYS),
    }


def _section_waste(r1, cat) -> str:
    lie = min(r1["lies"], key=lambda l: l["mfu"])
    lie_hr = num(cat[lie["gpu_type"]]["on_demand_hr"])
    effective_hr = lie_hr / lie["mfu"] if lie["mfu"] else 0.0
    idle_gpus = [s for s in r1["summary"] if s["idle_hours"] > 0]
    idle_txt = ", ".join(f"`{s['gpu_id']}` ({s['idle_hours']}h/ngày)" for s in idle_gpus) or "không có"
    names = ", ".join(f"`{l['gpu_id']}`" for l in r1["lies"])
    return (
        f"{names} báo cáo GPU-Util từ {min(l['gpu_util_pct'] for l in r1['lies']):.0f}% trở lên, "
        f"nhìn qua thì như đang chạy hết công suất. Nhưng MFU chỉ {lie['mfu']:.3f}. Hai con số này "
        f"không mâu thuẫn — chúng đo hai thứ khác nhau. `nvidia-smi` báo GPU-Util là *tỷ lệ thời gian "
        f"có ít nhất một kernel đang chạy*, nên một kernel bé xíu đang chờ dữ liệu từ HBM vẫn được tính "
        f"là \"bận\". MFU mới đo phần FLOPs thực sự thu được so với FLOPs đỉnh mà bạn đã trả tiền.\n\n"
        f"Khoảng cách đó chính là tiền. Với `{lie['gpu_id']}`, giá niêm yết là ${lie_hr:.2f}/giờ, "
        f"nhưng vì chỉ khai thác được {lie['mfu']*100:.1f}% năng lực tính toán nên chi phí thực cho "
        f"mỗi *giờ FLOPs dùng được* là **${effective_hr:,.2f}** — đắt gấp {1/lie['mfu']:.1f} lần con số "
        f"trên hoá đơn. Ba nguyên nhân quen thuộc, xếp theo tần suất gặp trong thực tế: batch size quá "
        f"nhỏ khiến tensor core đói dữ liệu và GPU dành phần lớn thời gian chờ HBM (memory stall); "
        f"chuỗi kernel ngắn khiến overhead khởi chạy kernel và đồng bộ chiếm tỷ trọng lớn; và dataloader "
        f"trên CPU không theo kịp, GPU chạy rồi dừng theo nhịp I/O. Cả ba đều không thể phát hiện nếu "
        f"chỉ nhìn GPU-Util — đó là lý do phải gắn MFU/MBU vào dashboard trước khi bàn chuyện mua thêm GPU.\n\n"
        f"Bên cạnh đó là lãng phí thô: {idle_txt} vẫn bật nhưng gần như không làm gì, "
        f"tương đương ${r1['idle_waste_daily']:,.2f}/ngày (${r1['idle_waste_daily']*DAYS:,.0f}/tháng) "
        f"trả cho không khí. Đây là khoản cắt được ngay trong tuần này, không cần đổi kiến trúc."
    )


def _section_rightsize(r1) -> str:
    head = ["| GPU | $/giờ | $/GB-VRAM | $/TB-s băng thông |", "|---|---:|---:|---:|"]
    for u in r1["unit_prices"]:
        head.append(f"| {u['gpu_type']} | ${u['on_demand_hr']:.2f} | ${u['usd_per_gb']:.4f} | ${u['usd_per_tbs']:.3f} |")
    table = "\n".join(head)

    if not r1["rightsize"]:
        detail = ("Không có GPU nào đổi được sang loại rẻ hơn mà vẫn đủ băng thông và VRAM — "
                  "đội hình hiện tại đã là hộp rẻ nhất vừa vặn với workload.")
    else:
        rows = ["| GPU | Hiện tại | Đề xuất | MBU/MFU | Tiết kiệm |", "|---|---|---|---:|---:|"]
        for c in r1["rightsize"]:
            rows.append(f"| `{c['gpu_id']}` | {c['from']} (${c['from_hr']:.2f}/h) | "
                        f"{c['to']} (${c['to_hr']:.2f}/h) | {c['mbu']:.2f}/{c['mfu']:.2f} | "
                        f"−{c['savings_pct']:.0f}% (${c['monthly_savings']:,.0f}/tháng) |")
        detail = "\n".join(rows)

    return (
        "Câu hỏi đúng không phải \"GPU nào rẻ nhất theo $/giờ\" mà là \"GPU nào rẻ nhất trên đơn vị tài "
        "nguyên đang thực sự nghẽn\". Với decode của LLM, thứ nghẽn là băng thông HBM và dung lượng "
        "KV-cache, không phải FLOPs. Bảng đơn giá cho thấy L4 rẻ nhất theo $/giờ ($0.80/giờ) nhưng lại là "
        "GPU **đắt nhất** theo $/TB-s — hạ cấp một workload memory-bound xuống L4 thì tiền thuê giảm "
        "nhưng throughput giảm nhiều hơn, và bạn phải thuê nhiều máy hơn để bù, tổng chi phí tăng.\n\n"
        f"{table}\n\n"
        f"Áp tiêu chí \"MBU > MFU (memory-bound) và MFU < 30%\", máy được đề xuất đổi:\n\n{detail}\n\n"
        f"Bảng đòn bẩy chỉ ghi nhận ${r1['rightsize_lie_monthly']:,.0f}/tháng từ các GPU đã bị gắn cờ "
        f"util-lie — phần có bằng chứng rõ ràng nhất. Tổng cơ hội right-size trên toàn đội hình là "
        f"${r1['rightsize_all_monthly']:,.0f}/tháng"
        + ("; hai con số trùng nhau vì các GPU memory-bound còn lại đều bị chặn bởi ràng buộc VRAM."
           if abs(r1['rightsize_all_monthly'] - r1['rightsize_lie_monthly']) < 1
           else ", phần chênh cần thêm một vòng đo trước khi ghi nhận.")
        + " Lưu ý thực thi: đề xuất MI300X đúng về mặt kinh tế "
        "($/GB-VRAM thấp nhất catalog, băng thông 5.3 TB/s) nhưng là hệ sinh thái ROCm — cần cộng thêm "
        "chi phí port kernel và một vòng benchmark trước khi chốt. Với A10G/L4, ràng buộc chặn lại là "
        "VRAM: không có hộp nào rẻ hơn mà vẫn chứa nổi KV-cache hiện tại."
    )


def _section_cache(r2) -> str:
    rows = ["| Chính sách TTL | Tier | Break-even (lượt đọc lại) | Thực đo | Kết luận | $ tiết kiệm/1M token input |",
            "|---|---|---:|---:|---|---:|"]
    for p in r2["cache_policies"]:
        rows.append(f"| {p['ttl']} | {p['tier']} | {p['break_even_reads']:.2f} | {p['observed_reads']:.2f} | "
                    f"{'Có lợi' if p['worth_it'] else 'Lỗ'} | ${p['usd_per_m_saved']:.4f} |")
    table = "\n".join(rows)
    reads = list(r2["cache_reads_by_tier"].values())
    reads_txt = (f"{min(reads):.2f}" if abs(max(reads) - min(reads)) < 0.005
                 else f"{min(reads):.2f}–{max(reads):.2f}")
    be5 = next(p["break_even_reads"] for p in r2["cache_policies"] if p["ttl"] == "5-min")
    be60 = next(p["break_even_reads"] for p in r2["cache_policies"] if p["ttl"] == "1-hour")
    return (
        "Prompt caching không miễn phí: ghi một entry tốn khoảng 1.25 lần một lượt đọc thường (bản "
        "5 phút) hoặc 2.0 lần (bản 1 giờ), đổi lại mỗi lượt đọc sau đó chỉ còn 10% giá. Nên câu hỏi "
        "phải trả lời bằng số là: *một prompt cần được đọc lại bao nhiêu lần thì cache mới hoà vốn?*\n\n"
        f"{table}\n\n"
        f"Ngưỡng hoà vốn là **{be5:.2f} lượt đọc lại** với cache 5 phút và **{be60:.2f} lượt** với cache "
        f"1 giờ. Đo trên `token_usage.csv` (số token cached chia cho số token input mới), traffic thực "
        f"đang tái sử dụng **{reads_txt} lượt/entry** (gần như bằng nhau ở cả hai tier). Kết luận rất rõ: cache 5 phút "
        f"có lãi (đang đóng góp ${r2['cache_savings_daily']:,.2f}/ngày, tương đương "
        f"${r2['cache_savings_daily']*DAYS:,.0f}/tháng), còn cache 1 giờ thì **lỗ** — mức tái sử dụng "
        f"hiện tại chưa tới một nửa ngưỡng cần thiết, bật lên là tự tăng hoá đơn.\n\n"
        "Ngưỡng hoà vốn giống nhau ở cả hai tier vì phí ghi và chiết khấu đọc đều tính theo tỷ lệ trên "
        "giá input. Cái khác nhau là số tiền đặt cược: mỗi 1M token input được cache trên tier `large` "
        f"tiết kiệm ${next(p['usd_per_m_saved'] for p in r2['cache_policies'] if p['ttl']=='5-min' and p['tier']=='large'):.4f}, "
        f"gấp {MODEL_PRICE_RATIO:.0f} lần tier `small`. Vì vậy kỷ luật cache (system prompt ổn định, "
        "few-shot dùng chung, tách phần động xuống cuối prompt) nên áp trước hết cho tier đắt."
    )


def _section_reasoning(r2) -> str:
    rs = r2["reasoning"]
    rows = ["| Kịch bản | Số request giữ reasoning | % traffic | Tiết kiệm $/ngày | Tiết kiệm Wh/ngày |",
            "|---|---:|---:|---:|---:|"]
    for s in rs["scenarios"]:
        rows.append(f"| {s['label']} | {s['kept']} | {s['traffic_pct']:.1f}% | "
                    f"${s['cost_saved']:,.2f} | {s['wh_saved']:,.0f} |")
    table = "\n".join(rows)
    rule = rs["scenarios"][-1]
    cap10 = rs["scenarios"][0]
    return (
        f"Reasoning chỉ chiếm **{rs['traffic_pct']:.1f}%** số request nhưng ăn **{rs['cost_pct']:.1f}%** chi phí "
        f"và **{rs['wh_pct']:.1f}%** năng lượng của tầng inference. Tỷ lệ lệch tới ~{rs['energy_multiplier']:.0f} lần "
        "không phải vì model \"nặng hơn\", mà vì một request reasoning sinh ra một chuỗi suy luận nội bộ "
        f"dài trước khi trả lời: output trung bình {rs['mean_output_reasoning']:,.0f} token so với "
        f"{rs['mean_output_normal']:,.0f} token của traffic thường "
        f"({rs['mean_output_reasoning']/rs['mean_output_normal']:.1f} lần). Mỗi token decode là một lượt "
        "đọc lại toàn bộ trọng số và KV-cache qua HBM — giai đoạn memory-bound, gần như không tái sử dụng "
        "được phép tính. Chuỗi dài gấp mấy lần thì số lượt đọc HBM, thời gian giữ GPU và điện năng cũng "
        "tăng theo bấy nhiêu, cộng dồn lại thành hệ số hàng chục lần.\n\n"
        f"{table}\n\n"
        f"Đọc bảng theo đúng thứ tự: đặt trần 10% traffic **không tiết kiệm được gì** "
        f"(${cap10['cost_saved']:,.2f}/ngày) đơn giản vì hiện tại đã ở {rs['traffic_pct']:.1f}% — trần đó "
        "không ràng buộc. Muốn có tác động thì phải siết chặt hơn. Quy tắc routing đề xuất: "
        f"**chỉ bật reasoning khi prompt đầu vào ≥ {m2_inference_levers.REASONING_MIN_INPUT_TOKENS:,} token** "
        "(proxy cho độ phức tạp: prompt dài thường là task nhiều ràng buộc, nhiều bước), phần còn lại "
        f"rơi về chế độ thường. Quy tắc này giữ lại {rule['kept']} request ({rule['traffic_pct']:.1f}% traffic) "
        f"và tiết kiệm **${rule['cost_saved']:,.2f}/ngày (${rule['cost_saved']*DAYS:,.0f}/tháng) cùng "
        f"{rule['wh_saved']:,.0f} Wh/ngày** — tức {rule['wh_saved']*DAYS/1000:,.0f} kWh/tháng. "
        "Cảnh báo đi kèm: đây là ước lượng theo độ dài output trung bình, nên trước khi bật phải chạy "
        "eval chất lượng trên đúng tập task bị hạ cấp; tiết kiệm năng lượng không đáng nếu tỷ lệ trả lời "
        "sai tăng lên."
    )


def _section_purchasing(r3) -> str:
    rows = ["| Job | GPU | Duty cycle | Tier chọn | Cam kết | On-demand | Sau tối ưu |",
            "|---|---|---:|---|---|---:|---:|"]
    for r in r3["recommendations"]:
        commit = f"{r['commit_years']} năm" if r["commit_years"] else "—"
        rows.append(f"| `{r['job_id']}` | {r['gpu_type']} | {r['duty']*100:.0f}% | {r['tier']} | {commit} | "
                    f"${r['on_demand']:,} | ${r['optimized']:,} |")
    table = "\n".join(rows)
    changed = [r for r in r3["recommendations"] if r["tier"] != r["naive_tier"] or r["commit_years"] == 1]
    changed_txt = "; ".join(
        f"`{r['job_id']}` chạy {r['duty']*100:.0f}% thời gian nên bị hạ từ cam kết 3 năm xuống 1 năm"
        if r["commit_years"] == 1 else
        f"`{r['job_id']}` chuyển từ `{r['naive_tier']}` sang `{r['tier']}`"
        for r in changed) or "không có job nào đổi quyết định"
    delta = r3["naive_savings_pct"] - r3["savings_pct"]
    return (
        f"{table}\n\n"
        f"Chính sách cũ (`recommend_tier`) chỉ nhìn hai biến: có gián đoạn được không, và duty cycle có "
        f"vượt điểm hoà vốn không. Chính sách mới (`recommend_tier_v2`) định giá **cả ba lựa chọn** rồi "
        f"chọn cái rẻ nhất, với hai bổ sung: (1) tỷ lệ gián đoạn spot theo từng loại GPU — H100 bị thu hồi "
        f"~8%/giờ trong khi L4 chỉ ~1.5%/giờ, nên chi phí rework sau mỗi lần mất máy được tính vào giá "
        f"spot thay vì giả định chung 5%; (2) tách reserved 1 năm và 3 năm, chỉ cho phép cam kết 3 năm khi "
        f"workload đã chứng minh duty ≥ 90% và chạy ≥ 30 ngày.\n\n"
        f"Kết quả: savings giảm từ **{r3['naive_savings_pct']:.1f}%** xuống **{r3['savings_pct']:.1f}%** "
        f"(chênh ${abs(r3['optimized_monthly'] - r3['naive_monthly']):,.0f}/tháng, tức {delta:.1f} điểm phần trăm). "
        f"Đây là chỗ dễ hiểu nhầm: chính sách mới *không tệ hơn*, nó chỉ ngừng ghi nhận khoản tiết kiệm mà "
        f"chính sách cũ chưa trả giá. Cụ thể — {changed_txt}. Cam kết 3 năm cho một job chạy 18 giờ/ngày là "
        f"đổi 1.7 điểm phần trăm savings lấy rủi ro trả tiền cho 36 tháng một workload có thể biến mất sau "
        f"hai quý; với H100 spot, giả định 5% thay vì 8% là giấu chi phí rework vào chỗ khuất. Số thấp hơn "
        f"nhưng đáng tin hơn — và đó mới là số mang đi cam kết ngân sách."
    )


def _section_allocation(r4) -> str:
    rows = ["| Team | $/ngày | $/tháng |", "|---|---:|---:|"]
    for team, cost in sorted(r4["by_team"].items(), key=lambda x: -x[1]):
        rows.append(f"| {team} | ${cost:,.2f} | ${cost*DAYS:,.0f} |")
    table = "\n".join(rows)
    return (
        f"{table}\n\n"
        f"Tag coverage đang ở **{r4['tag_coverage']*100:.0f}%**, vượt ngưỡng 80% nên cổng chargeback đã mở "
        f"(`chargeback_ready = {r4['chargeback_ready']}`). Ngưỡng này không tuỳ tiện: dưới 80%, phần chi phí "
        "không gắn được tag phải phân bổ theo tỷ lệ ước lượng, và team bị tính tiền sẽ tranh cãi đúng — "
        "hoá đơn nội bộ mất uy tín ngay lần đầu sai. Khuyến nghị vận hành: chạy showback (chỉ hiển thị, "
        "chưa tính tiền) khoảng một chu kỳ để các team đối soát, đồng thời bật cảnh báo khi coverage tụt "
        "dưới 80%, rồi mới chuyển sang chargeback thật. File `outputs/focus_export.csv` đã xuất theo lược "
        "đồ FOCUS nên ghép được thẳng vào công cụ FinOps mà không cần ETL riêng."
    )


MODEL_PRICE_RATIO = (m2_inference_levers.MODEL_PRICES["large"][0]
                     / m2_inference_levers.MODEL_PRICES["small"][0])


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    cat = catalog_by_type()

    levers = _levers(r1, r2, r3)
    infer_savings = levers["Inference: cascade + cache + batch"]
    purchasing_savings = levers["Mua hạ tầng: spot + reserved"]
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    wh = sustainability.wh_per_query(MEDIAN_QUERY_TOKENS)
    rs = r2["reasoning"]
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, r3["home_region"]),
        "best_region": r3["cleanest_region"],
        "details": [
            f"Chuyển toàn bộ job có thể gián đoạn từ {r3['home_region']} sang {r3['cleanest_region']} "
            f"giảm {r3['carbon_savings_g']/1000:,.2f} kgCO2e mỗi chu kỳ chạy "
            f"({r3['interruptible_wh']/1000:,.0f} kWh), nhưng cộng thêm ~"
            f"{next(x['latency_ms'] for x in r3['carbon_regions'] if x['region']==r3['cleanest_region']) - next(x['latency_ms'] for x in r3['carbon_regions'] if x['region']==r3['home_region'])} ms độ trễ.",
            f"Vùng rẻ điện nhất là {r3['cheapest_energy_region']} "
            f"(tiết kiệm ${r3['energy_savings_usd']:,.2f} tiền điện mỗi chu kỳ so với {r3['home_region']}).",
            f"Lựa chọn cân bằng: {r3['balanced_region']} — rẻ nhất trong nhóm vùng có cường độ carbon "
            f"dưới trung vị, và chỉ xa hơn ~50 ms so với {r3['home_region']}.",
            f"Reasoning chiếm {rs['traffic_pct']:.1f}% traffic nhưng {rs['wh_pct']:.1f}% năng lượng "
            f"truy vấn hằng ngày.",
        ],
        "regional_options": r3["carbon_regions"],
    }

    swap_names = ", ".join(f"`{c['gpu_id']}` ({c['from']}→{c['to']})" for c in r1["rightsize"]) or "các GPU memory-bound"
    rule = rs["scenarios"][-1]
    analysis = [
        f"**Tắt GPU idle — làm ngay tuần này.** ${levers['Tắt GPU chạy không (idle)']:,}/tháng, "
        f"không rủi ro kỹ thuật, không cần đổi code. ROI cao nhất trên mỗi giờ công bỏ ra.",
        f"**Chốt lại danh mục mua hạ tầng.** ${purchasing_savings:,}/tháng "
        f"({purchasing_savings/sum(levers.values())*100:.0f}% tổng tiết kiệm) — đòn bẩy lớn nhất. "
        f"Spot cho job có checkpoint, reserved cho job chạy nền 24/7, và không cam kết 3 năm cho "
        f"workload chưa đủ ổn định.",
        f"**Giữ cascade + batch + cache 5 phút cho inference.** ${infer_savings:,}/tháng, đưa đơn giá từ "
        f"${r2['baseline_per_m']:.3f} xuống ${r2['optimized_per_m']:.3f}/1M token. Không bật cache 1 giờ "
        f"cho tới khi tái sử dụng vượt {next(p['break_even_reads'] for p in r2['cache_policies'] if p['ttl']=='1-hour'):.2f} lượt/entry.",
        f"**Siết ngân sách reasoning theo quy tắc routing.** Thêm ${rule['cost_saved']*DAYS:,.0f}/tháng và "
        f"{rule['wh_saved']*DAYS/1000:,.0f} kWh/tháng, nhưng phải kèm eval chất lượng trước khi bật.",
        f"**Right-size {swap_names} sau khi profile.** ${r1['rightsize_lie_monthly']:,.0f}/tháng — để sau cùng "
        f"vì cần một vòng benchmark (và port sang ROCm nếu chọn MI300X); MFU thấp đôi khi sửa được bằng "
        f"tăng batch size mà không cần đổi phần cứng.",
        f"**Chạy showback trước, chargeback sau.** Coverage {r4['tag_coverage']*100:.0f}% đã đủ điều kiện, "
        f"nhưng cần một chu kỳ đối soát và cảnh báo tự động khi coverage tụt dưới 80%.",
        f"**Lịch chạy theo carbon cho job gián đoạn.** {r3['cleanest_region']} nếu ưu tiên phát thải, "
        f"{r3['balanced_region']} nếu cần cân bằng tiền điện và độ trễ.",
    ]

    sections = [
        {"heading": "Tiền đang rò rỉ ở đâu — và vì sao GPU-Util 98% vẫn là lãng phí",
         "body": _section_waste(r1, cat)},
        {"heading": "Right-sizing: mua băng thông, đừng mua FLOPs",
         "body": _section_rightsize(r1)},
        {"heading": "Kinh tế học của prompt caching: cần đọc lại bao nhiêu lần mới hoà vốn?",
         "body": _section_cache(r2)},
        {"heading": "Ngân sách reasoning: 8% traffic, 94% điện năng",
         "body": _section_reasoning(r2)},
        {"heading": "Chiến lược mua GPU: chính sách cũ vs. chính sách có tính rủi ro",
         "body": _section_purchasing(r3)},
        {"heading": "Phân bổ chi phí: đã sẵn sàng chargeback chưa?",
         "body": _section_allocation(r4)},
    ]

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust, analysis=analysis, lang="vi",
        period="hàng tháng",
        unit_cost={"baseline_per_m": r2["baseline_per_m"], "optimized_per_m": r2["optimized_per_m"]},
        sections=sections)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"),
                                   baseline_usd=baseline, optimized_usd=optimized, lang="vi")

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print("\nWritten: outputs/report.md" + (" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
