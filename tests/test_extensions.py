"""Tests for the five "Your Turn" extensions (student-written).

These cover the new logic only — the graded 15 tests are untouched.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finops import metrics, pricing, report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

CATALOG = {
    "H100": {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0,
             "reserved_3yr_hr": 1.4, "hbm_gb": 80, "peak_bw_tbs": 3.35},
    "A100": {"on_demand_hr": 1.79, "spot_hr": 1.1, "reserved_1yr_hr": 1.4,
             "reserved_3yr_hr": 1.0, "hbm_gb": 80, "peak_bw_tbs": 2.0},
    "L4": {"on_demand_hr": 0.8, "spot_hr": 0.35, "reserved_1yr_hr": 0.6,
           "reserved_3yr_hr": 0.45, "hbm_gb": 24, "peak_bw_tbs": 0.3},
}


# --- #1 risk-adjusted purchasing ---------------------------------------------
def test_v2_prefers_reserved_for_steady_and_commits_three_years():
    pick = pricing.recommend_tier_v2(24, False, 24 * 30, CATALOG["A100"], gpu_type="A100", days=30)
    assert pick["tier"] == "reserved" and pick["commit_years"] == 3


def test_v2_refuses_three_year_lockin_below_duty_threshold():
    pick = pricing.recommend_tier_v2(18, False, 18 * 30, CATALOG["L4"], gpu_type="L4", days=30)
    assert pick["tier"] == "reserved" and pick["commit_years"] == 1


def test_v2_spot_price_reflects_gpu_specific_interruption_rate():
    hours = 20 * 30
    risky = pricing.recommend_tier_v2(20, True, hours, CATALOG["H100"], gpu_type="H100", days=14)
    safe = pricing.recommend_tier_v2(20, True, hours, CATALOG["H100"], gpu_type="H100",
                                     days=14, interrupt_rate=0.0)
    assert risky["options"]["spot"] > safe["options"]["spot"]   # rework is priced in
    assert risky["tier"] == "spot"                              # spot still wins


def test_v2_falls_back_to_on_demand_for_spiky_workloads():
    pick = pricing.recommend_tier_v2(2, False, 2 * 30, CATALOG["H100"], gpu_type="H100", days=5)
    assert pick["tier"] == "on_demand"


# --- #2 MBU-driven right-sizing ----------------------------------------------
def test_unit_prices_rank_bandwidth_not_sticker_price():
    # L4 is the cheapest box per hour but the most expensive per TB/s.
    assert metrics.dollars_per_tbs(0.8, 0.3) > metrics.dollars_per_tbs(2.5, 3.35)
    assert metrics.is_memory_bound(mfu=0.19, mbu=0.21)
    assert not metrics.is_memory_bound(mfu=0.42, mbu=0.42)


def test_rightsize_candidate_respects_bandwidth_and_vram_headroom():
    swap = metrics.rightsize_candidate("H100", CATALOG, achieved_bw_tbs=1.0, vram_needed_gb=40)
    assert swap["to"] == "A100" and swap["hourly_savings"] > 0
    # A workload that needs more VRAM than any cheaper box has stays put.
    assert metrics.rightsize_candidate("H100", CATALOG, achieved_bw_tbs=1.0, vram_needed_gb=75) is None


# --- #3 cache economics -------------------------------------------------------
def test_cache_break_even_rises_with_the_write_premium():
    cheap = pricing.cache_break_even_reads(pricing.CACHE_TTL_POLICIES["5-min"], 0.10)
    extended = pricing.cache_break_even_reads(pricing.CACHE_TTL_POLICIES["1-hour"], 0.10)
    assert cheap < extended
    assert abs(cheap - (0.25 / 0.90)) < 1e-9


def test_cache_verdict_flips_between_ttl_policies_at_observed_reuse():
    reads = 0.47   # measured in token_usage.csv
    assert pricing.cache_is_worth_it(reads, pricing.CACHE_TTL_POLICIES["5-min"], 0.10)
    assert not pricing.cache_is_worth_it(reads, pricing.CACHE_TTL_POLICIES["1-hour"], 0.10)
    assert pricing.cache_savings_per_million(3.0, reads, 2.0, 0.10) < 0


def test_observed_cache_reads_measured_from_traffic():
    rows = [{"route_tier": "small", "input_tokens": "1000", "cached_input_tokens": "500"},
            {"route_tier": "large", "input_tokens": "1000", "cached_input_tokens": "0"}]
    reads = m2_inference_levers.observed_cache_reads(rows)
    assert abs(reads["small"] - 1.0) < 1e-9    # 500 cached per 500 fresh
    assert reads["large"] == 0.0


# --- #4 reasoning budget ------------------------------------------------------
def test_reasoning_cap_cuts_cost_and_energy_together():
    r2 = m2_inference_levers.run(verbose=False)
    rs = r2["reasoning"]
    assert rs["wh_pct"] > rs["traffic_pct"]          # energy share dwarfs traffic share
    rule = rs["scenarios"][-1]
    assert rule["cost_saved"] > 0 and rule["wh_saved"] > 0
    assert rule["traffic_pct"] < rs["traffic_pct"]


# --- #5 carbon-aware scheduling ----------------------------------------------
def test_region_scorecard_separates_cleanest_cheapest_and_latency():
    regions = sustainability.region_scorecard(1_000)
    assert min(regions, key=lambda r: r["carbon_g"])["region"] == "europe-north1"
    assert min(regions, key=lambda r: r["energy_cost_usd"])["region"] == "us-east-wa"
    assert min(regions, key=lambda r: r["latency_ms"])["region"] == "us-east-1"


def test_m3_reports_carbon_savings_for_interruptible_jobs():
    r3 = m3_purchasing.run(verbose=False)
    assert r3["carbon_savings_g"] > 0 and r3["energy_savings_usd"] > 0
    assert r3["naive_savings_pct"] >= r3["savings_pct"]   # risk-adjusted is conservative


# --- report rendering ---------------------------------------------------------
def test_vietnamese_report_carries_unit_cost_and_sections():
    md = report.build_report(1000, 600, {"cache": 400}, lang="vi",
                             unit_cost={"baseline_per_m": 6.5, "optimized_per_m": 1.1},
                             sections=[{"heading": "Phân tích", "body": "nội dung"}])
    assert "Tiết kiệm dự kiến" in md and "$/1M token" in md and "Phân tích" in md
