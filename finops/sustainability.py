"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


def region_comparison(wh: float) -> list[dict]:
    """Return comparable electricity cost and carbon for every supported region."""
    return [
        {
            "region": region,
            "energy_cost_usd": energy_cost_usd(wh, region),
            "carbon_g": carbon_g(wh, region),
            "price_kwh": REGION_PRICE_KWH[region],
            "carbon_kwh": REGION_CARBON[region],
        }
        for region in REGION_CARBON
    ]


# Illustrative round-trip latency (ms) from a US-East user base — the price you
# pay for scheduling in the cleanest grid.
REGION_LATENCY_MS = {
    "us-east-1": 15,
    "us-west-2": 70,
    "europe-north1": 110,
    "europe-central2": 120,
    "us-east-wa": 65,
}


def region_scorecard(wh: float) -> list[dict]:
    """Region comparison enriched with latency, for the cost/carbon/latency trade-off."""
    rows = region_comparison(wh)
    for row in rows:
        row["latency_ms"] = REGION_LATENCY_MS.get(row["region"], 0)
    return rows
