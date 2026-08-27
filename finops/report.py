"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


LABELS = {
    "en": {
        "title": "NimbusAI — GPU Cost Optimization Report",
        "period": "Period", "baseline": "Baseline spend", "optimized": "Optimized spend",
        "savings": "Projected savings", "unit": "Unit cost ($/1M tokens)",
        "levers": "Savings by lever", "lever": "Lever", "amount": "Savings (USD)", "share": "Share",
        "sustainability": "Sustainability",
        "wh": "Energy per query", "carbon": "Carbon per query", "region": "Cleanest region",
        "regional": "Carbon-aware regional options",
        "region_col": "Region", "elec": "Electricity cost", "carbon_col": "Carbon", "latency": "Latency",
        "analysis": "Findings and prioritized actions",
        "footer": "_Figures are June-2026 as-of snapshots; re-baseline before acting._",
    },
    "vi": {
        "title": "NimbusAI — Báo cáo tối ưu chi phí GPU",
        "period": "Kỳ báo cáo", "baseline": "Chi tiêu hiện tại (baseline)",
        "optimized": "Chi tiêu sau tối ưu", "savings": "Tiết kiệm dự kiến",
        "unit": "Đơn giá ($/1M token)",
        "levers": "Tiết kiệm theo từng đòn bẩy", "lever": "Đòn bẩy",
        "amount": "Tiết kiệm (USD)", "share": "Tỷ trọng",
        "sustainability": "Tính bền vững",
        "wh": "Năng lượng mỗi truy vấn", "carbon": "Carbon mỗi truy vấn",
        "region": "Vùng sạch nhất",
        "regional": "So sánh các vùng triển khai",
        "region_col": "Vùng", "elec": "Tiền điện", "carbon_col": "Carbon", "latency": "Độ trễ",
        "analysis": "Kết luận và hành động theo thứ tự ưu tiên",
        "footer": "_Số liệu là ảnh chụp giá tháng 6/2026 — hãy dựng lại baseline trước khi áp dụng thực tế._",
    },
}


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 analysis: list[str] | None = None, lang: str = "en",
                 unit_cost: dict | None = None, sections: list[dict] | None = None) -> str:
    """Return a markdown cost-optimization report.

    ``lang`` switches the fixed labels ("en" keeps the original wording).
    ``unit_cost`` promotes the $/1M-token before/after to the summary block, and
    ``sections`` lets a mission append its own prose analysis between the lever
    table and the sustainability block.
    """
    L = LABELS.get(lang, LABELS["en"])
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    total_levers = sum(levers.values()) or 1.0
    lines = [
        f"# {L['title']}",
        "",
        f"**{L['period']}:** {period}  ",
        f"**{L['baseline']}:** ${baseline_usd:,.0f}  ",
        f"**{L['optimized']}:** ${optimized_usd:,.0f}  ",
        (f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)"
         if lang == "en" else
         f"**{L['savings']}:** ${savings:,.0f}  (**{pct:.0f}%**)") + "  ",
    ]
    if unit_cost:
        lines.append(
            f"**{L['unit']}:** ${unit_cost['baseline_per_m']:.3f} → "
            f"**${unit_cost['optimized_per_m']:.3f}**  "
            f"(−{(1 - unit_cost['optimized_per_m'] / unit_cost['baseline_per_m']) * 100:.0f}%)"
        )
    lines += [
        "",
        f"## {L['levers']}",
        "",
        f"| {L['lever']} | {L['amount']} | {L['share']} |",
        "|---|---:|---:|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} | {amount / total_levers * 100:.0f}% |")
    for section in (sections or []):
        lines += ["", f"## {section['heading']}", "", section["body"]]
    if sustainability:
        lines += [
            "",
            f"## {L['sustainability']}",
            "",
            f"- {L['wh']}: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- {L['carbon']}: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- {L['region']}: {sustainability.get('best_region', 'n/a')}",
        ]
        for detail in sustainability.get("details", []):
            lines.append(f"- {detail}")
        regional_options = sustainability.get("regional_options", [])
        if regional_options:
            lines += ["", f"### {L['regional']}", "",
                      f"| {L['region_col']} | $/kWh | gCO2/kWh | {L['elec']} | {L['carbon_col']} | {L['latency']} |",
                      "|---|---:|---:|---:|---:|---:|"]
            for option in regional_options:
                lines.append(
                    f"| {option['region']} | {option.get('price_kwh', 0):.3f} | {option.get('carbon_kwh', 0)} | "
                    f"${option['energy_cost_usd']:,.2f} | "
                    f"{option['carbon_g'] / 1000:,.2f} kgCO2e | {option.get('latency_ms', 0)} ms |"
                )
    if analysis:
        lines += ["", f"## {L['analysis']}", ""]
        lines.extend(f"{i}. {item}" for i, item in enumerate(analysis, start=1))
    lines += ["", L["footer"]]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline_usd: float | None = None,
                      optimized_usd: float | None = None, lang: str = "en") -> str:
    """Write the savings chart PNG. Returns the path. No-op if matplotlib absent.

    With ``baseline_usd``/``optimized_usd`` it draws a true waterfall: the
    starting bill, one falling step per lever, and the remaining bill.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    vi = lang == "vi"
    names = list(levers.keys())
    vals = [levers[n] for n in names]

    if baseline_usd is None or optimized_usd is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
        ax.set_title("GPU cost savings by FinOps lever")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    start_label = "Chi phí hiện tại" if vi else "Baseline"
    end_label = "Sau tối ưu" if vi else "Optimized"
    labels = [start_label] + names + [end_label]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    # matplotlib reads a bare "$" as mathtext, so every label escapes it.
    money = lambda v: f"\\${v:,.0f}"

    running = baseline_usd
    ax.bar(0, baseline_usd, color="#8c3b3b")
    ax.text(0, baseline_usd, money(baseline_usd), ha="center", va="bottom", fontsize=9)
    small = 0.10 * baseline_usd   # thin bars get their label outside, in dark ink
    for i, (name, val) in enumerate(zip(names, vals), start=1):
        ax.bar(i, -val, bottom=running, color="#2e548a")
        if val >= small:
            ax.text(i, running - val / 2, "-" + money(val), ha="center", va="center",
                    fontsize=9, color="white")
        else:
            ax.text(i, running - val - baseline_usd * 0.02, "-" + money(val), ha="center",
                    va="top", fontsize=9, color="#2e548a")
        running -= val
    ax.bar(len(names) + 1, optimized_usd, color="#2f6b4f")
    ax.text(len(names) + 1, optimized_usd, money(optimized_usd), ha="center",
            va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("USD / tháng" if vi else "USD / month")
    saved_pct = (baseline_usd - optimized_usd) / baseline_usd * 100 if baseline_usd else 0
    ax.set_title(
        (f"Chi phí GPU hàng tháng: {money(baseline_usd)} → {money(optimized_usd)} "
         f"(tiết kiệm {saved_pct:.0f}%)") if vi else
        (f"Monthly GPU spend: {money(baseline_usd)} -> {money(optimized_usd)} "
         f"({saved_pct:.0f}% saved)"))
    ax.set_ylim(0, baseline_usd * 1.10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
