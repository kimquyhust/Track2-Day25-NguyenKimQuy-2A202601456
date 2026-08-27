"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Also implements Your Turn #2: MBU-driven right-sizing priced in $/GB-VRAM and
$/TB-s instead of the headline $/GPU-hr.

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
LOW_MFU = 0.30  # below this a GPU is not earning the FLOPs you rented


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None,
                               "idle_hours": 0, "bw": [], "mem": []})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["mem"].append(num(r["mem_used_gb"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "peak_bw_tbs": max(a["bw"]),      # p100 so the swap keeps headroom
            "peak_mem_gb": max(a["mem"]),
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(cat[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Your Turn #2: right-size the memory-bound, low-MFU boxes -------------
    lie_ids = {l["gpu_id"] for l in lies}
    rightsize = []
    for s in summary:
        s["memory_bound"] = metrics.is_memory_bound(s["mfu"], s["mbu"])
        if not (s["memory_bound"] and s["mfu"] < LOW_MFU):
            continue
        cand = metrics.rightsize_candidate(s["gpu_type"], cat, s["peak_bw_tbs"], s["peak_mem_gb"])
        if not cand:
            continue
        cand["gpu_id"] = s["gpu_id"]
        cand["mfu"] = s["mfu"]
        cand["mbu"] = s["mbu"]
        cand["monthly_savings"] = cand["hourly_savings"] * 24 * DAYS
        cand["is_util_lie"] = s["gpu_id"] in lie_ids
        rightsize.append(cand)

    # The headline lever only counts the util-lie GPUs (the defensible, already
    # evidenced ones).  The rest is reported as an opportunity, not booked.
    lie_rightsize_monthly = sum(c["monthly_savings"] for c in rightsize if c["is_util_lie"])
    all_rightsize_monthly = sum(c["monthly_savings"] for c in rightsize)

    unit_prices = sorted(
        ({"gpu_type": g,
          "on_demand_hr": num(row["on_demand_hr"]),
          "usd_per_gb": metrics.dollars_per_gb_vram(num(row["on_demand_hr"]), num(row["hbm_gb"])),
          "usd_per_tbs": metrics.dollars_per_tbs(num(row["on_demand_hr"]), num(row["peak_bw_tbs"])),
          "hbm_gb": num(row["hbm_gb"]), "peak_bw_tbs": num(row["peak_bw_tbs"])}
         for g, row in cat.items()),
        key=lambda x: x["usd_per_tbs"])

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}{'bound':>15}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}"
                  f"{s['mbu']:>7}{s['idle_hours']:>8}"
                  f"{'memory-bound' if s['memory_bound'] else 'compute-bound':>15}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*DAYS:,.0f}/month")

        print("\n-- unit economics (cheapest bandwidth first) --")
        print(f"{'gpu':8}{'$/hr':>8}{'$/GB-VRAM':>12}{'$/TB-s':>10}")
        for u in unit_prices:
            print(f"{u['gpu_type']:8}{u['on_demand_hr']:>8.2f}{u['usd_per_gb']:>12.4f}{u['usd_per_tbs']:>10.3f}")

        print("\n-- right-sizing (memory-bound & MFU<30%) --")
        if not rightsize:
            print("  nothing to swap: every memory-bound GPU is already the cheapest box that fits")
        for c in rightsize:
            print(f"  {c['gpu_id']:14}{c['from']:>6} -> {c['to']:<6} "
                  f"${c['from_hr']:.2f}/hr -> ${c['to_hr']:.2f}/hr  (-{c['savings_pct']:.0f}%, "
                  f"${c['monthly_savings']:,.0f}/mo)  MBU {c['mbu']:.2f} > MFU {c['mfu']:.2f}"
                  f"{'  [util-lie]' if c['is_util_lie'] else ''}")
        print(f"  booked lever (util-lies only): ${lie_rightsize_monthly:,.0f}/month")
        print(f"  full opportunity (all memory-bound): ${all_rightsize_monthly:,.0f}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsize": rightsize,
            "rightsize_lie_monthly": round(lie_rightsize_monthly, 2),
            "rightsize_all_monthly": round(all_rightsize_monthly, 2),
            "unit_prices": unit_prices}


if __name__ == "__main__":
    run()
