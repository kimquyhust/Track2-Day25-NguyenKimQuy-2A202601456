"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Also implements Your Turn #1 (risk-adjusted tier policy, benchmarked against the
naive one) and Your Turn #5 (carbon-aware scheduling for interruptible jobs).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30
HOME_REGION = "us-east-1"


def _naive_cost(job, cat, gpu_hours) -> tuple[str, float]:
    """The lab's original policy, kept as the benchmark to beat."""
    c = cat[job["gpu_type"]]
    od = num(c["on_demand_hr"])
    tier = pricing.recommend_tier(num(job["hours_per_day"]), bool(int(num(job["interruptible"]))))
    if tier == "spot":
        return tier, pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
    if tier == "reserved":
        return tier, gpu_hours * num(c["reserved_3yr_hr"])
    return tier, gpu_hours * od


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = naive_monthly = 0.0
    recs = []
    interruptible_wh = 0.0
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        # Use the workload's actual duration for the carbon scheduling analysis.
        scheduled_gpu_hours = hpd * num(j["days"]) * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        naive_tier, naive_cost = _naive_cost(j, cat, gpu_hours)
        pick = pricing.recommend_tier_v2(hpd, interruptible, gpu_hours, c,
                                         gpu_type=gtype, days=num(j["days"]))
        tier, opt_cost = pick["tier"], pick["cost"]

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        naive_monthly += naive_cost
        if interruptible:
            interruptible_wh += scheduled_gpu_hours * num(c["watts"])
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "commit_years": pick["commit_years"], "reason": pick["reason"],
                     "duty": round(pick["duty"], 2),
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
                     "naive_tier": naive_tier, "naive": round(naive_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    naive_pct = (on_demand_monthly - naive_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- Your Turn #5: carbon-aware scheduling for interruptible jobs ---------
    carbon_regions = sustainability.region_scorecard(interruptible_wh)
    current_region = next(r for r in carbon_regions if r["region"] == HOME_REGION)
    cleanest_region = min(carbon_regions, key=lambda r: r["carbon_g"])
    cheapest_region = min(carbon_regions, key=lambda r: r["energy_cost_usd"])
    # Balanced = cheapest electricity among regions at or below the median grid
    # intensity, i.e. don't buy cheap power from the dirtiest grid.
    median_carbon = sorted(r["carbon_kwh"] for r in carbon_regions)[len(carbon_regions) // 2]
    balanced_region = min((r for r in carbon_regions if r["carbon_kwh"] <= median_carbon),
                          key=lambda r: r["energy_cost_usd"] + r["carbon_g"] / 1e6)
    carbon_savings_g = current_region["carbon_g"] - cleanest_region["carbon_g"]
    energy_savings_usd = current_region["energy_cost_usd"] - cheapest_region["energy_cost_usd"]

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'duty':>6}  {'tier':10}{'commit':>7}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            commit = f"{r['commit_years']}yr" if r["commit_years"] else "-"
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['duty']:>6.2f}  {r['tier']:10}{commit:>7}"
                  f"${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"policy benchmark: naive policy {naive_pct:.1f}% vs risk-adjusted {savings_pct:.1f}% "
              f"(difference ${optimized_monthly - naive_monthly:,.0f}/month)")
        for r in recs:
            if r["tier"] != r["naive_tier"] or r["commit_years"] == 1:
                print(f"  changed: {r['job_id']:18}{r['naive_tier']} -> {r['tier']}"
                      f"{'/' + str(r['commit_years']) + 'yr' if r['commit_years'] else ''}  ({r['reason']})")

        print("\ncarbon-aware scheduling (interruptible jobs, "
              f"{interruptible_wh/1000:,.0f} kWh per scheduled run):")
        print(f"  {'region':16}{'$/kWh':>8}{'gCO2/kWh':>10}{'electricity':>13}{'carbon':>14}{'latency':>9}")
        for reg in carbon_regions:
            print(f"  {reg['region']:16}{reg['price_kwh']:>8.3f}{reg['carbon_kwh']:>10}"
                  f"${reg['energy_cost_usd']:>12,.2f}{reg['carbon_g']/1000:>11,.2f} kg{reg['latency_ms']:>7} ms")
        print(f"  cleanest: {cleanest_region['region']} (-{carbon_savings_g/1000:,.2f} kgCO2e vs {HOME_REGION})")
        print(f"  cheapest electricity: {cheapest_region['region']} (-${energy_savings_usd:,.2f} vs {HOME_REGION})")
        print(f"  balanced pick: {balanced_region['region']}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "naive_monthly": round(naive_monthly), "naive_savings_pct": round(naive_pct, 1),
            "interruptible_wh": round(interruptible_wh, 2),
            "carbon_regions": carbon_regions,
            "home_region": HOME_REGION,
            "cleanest_region": cleanest_region["region"],
            "cheapest_energy_region": cheapest_region["region"],
            "balanced_region": balanced_region["region"],
            "carbon_savings_g": round(carbon_savings_g, 2),
            "energy_savings_usd": round(energy_savings_usd, 2)}


if __name__ == "__main__":
    run()
