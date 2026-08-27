"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def cache_is_worth_it(
    avg_reads: float,
    write_cost: float = 1.25,
    read_discount: float = 0.10,
) -> bool:
    """Whether caching is cheaper than serving the same prompt uncached.

    Costs are expressed relative to one ordinary input read.  Creating a cache
    entry can cost more than that read (``write_cost``); each later cache read
    costs ``read_discount``.  The comparison includes the initial ordinary
    request, so it deliberately answers the practical question: *will this
    prompt be reused often enough to recover its cache-write cost?*
    """
    reads = max(0.0, float(avg_reads))
    write = max(0.0, float(write_cost))
    discount = max(0.0, min(1.0, float(read_discount)))
    cached_total = write + reads * discount
    uncached_total = 1.0 + reads
    return cached_total < uncached_total


def cache_break_even_reads(write_cost: float = 1.25, read_discount: float = 0.10) -> float:
    """Minimum additional reads required for a cache entry to pay for itself."""
    write = max(0.0, float(write_cost))
    discount = max(0.0, min(1.0, float(read_discount)))
    if discount >= 1.0:
        return float("inf")
    return max(0.0, (write - 1.0) / (1.0 - discount))


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# --- Your Turn #3: cache economics -------------------------------------------
# Two cache TTL policies with different write premiums.  A 5-minute entry is
# cheap to create; an extended (1-hour) entry costs roughly twice a normal read
# to write, so it needs far more reuse before it pays for itself.
CACHE_TTL_POLICIES = {
    "5-min": 1.25,
    "1-hour": 2.00,
}


def cache_savings_per_million(
    price_in_per_m: float,
    avg_reads: float,
    write_cost: float = 1.25,
    read_discount: float = 0.10,
) -> float:
    """USD saved per 1M input tokens of a prompt that is re-read ``avg_reads`` times.

    The break-even *read count* is price-independent, so both model tiers flip at
    the same threshold — but the dollars at stake scale with the tier's input
    price, which is why caching discipline matters most on the expensive tier.
    """
    reads = max(0.0, float(avg_reads))
    write = max(0.0, float(write_cost))
    discount = max(0.0, min(1.0, float(read_discount)))
    uncached_units = 1.0 + reads
    cached_units = write + reads * discount
    return (uncached_units - cached_units) * float(price_in_per_m)


# --- Your Turn #1: risk-adjusted purchasing policy ---------------------------
# Per-hour spot interruption rate by GPU type.  Scarce top-end accelerators get
# reclaimed far more often than commodity inference cards.
SPOT_INTERRUPT_RATE = {
    "B200": 0.12,
    "H200": 0.10,
    "H100": 0.08,
    "MI300X": 0.06,
    "A100": 0.05,
    "A10G": 0.02,
    "L4": 0.015,
}
# A 3-year lock-in is only defensible for a workload that is already proven
# steady; anything below these thresholds commits for 1 year instead.
COMMIT_3YR_MIN_DUTY = 0.90
COMMIT_3YR_MIN_DAYS = 30


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    gpu_hours: float,
    prices: dict,
    gpu_type: str | None = None,
    days: float = 30,
    interrupt_rate: float | None = None,
) -> dict:
    """Risk-adjusted tier choice: prices every option, then picks the cheapest.

    Improves on :func:`recommend_tier` in two ways:

    1. Spot is priced through :func:`spot_checkpoint_cost` with a *per-GPU-type*
       interruption rate, so an H100 (reclaimed ~8%/h) is not assumed to be as
       safe as an L4 (~1.5%/h).
    2. Reserved is split into 1-year and 3-year commitments.  The 3-year rate is
       only offered to workloads that clear both a duty-cycle and a duration
       threshold; everything else pays the shallower 1-year discount rather than
       locking a 36-month bill to a workload that may not exist next quarter.

    Returns the tier, the commitment length, the modelled cost and a reason.
    """
    duty = max(0.0, hours_per_day) / 24.0
    od_hr = float(prices["on_demand_hr"])
    options = {"on_demand": {"cost": gpu_hours * od_hr, "commit_years": 0,
                             "reason": "duty cycle too low or too risky to commit"}}

    if interruptible:
        rate = SPOT_INTERRUPT_RATE.get(gpu_type, 0.05) if interrupt_rate is None else interrupt_rate
        sim = spot_checkpoint_cost(gpu_hours, float(prices["spot_hr"]), od_hr, interrupt_rate=rate)
        options["spot"] = {
            "cost": sim["spot_cost"], "commit_years": 0,
            "reason": f"checkpointable; {rate:.1%}/h interruption priced in "
                      f"({sim['spot_effective_hours']:.0f} effective GPU-h)",
        }

    steady_enough = duty >= COMMIT_3YR_MIN_DUTY and days >= COMMIT_3YR_MIN_DAYS
    years = 3 if steady_enough else 1
    res_hr = float(prices["reserved_3yr_hr"] if years == 3 else prices["reserved_1yr_hr"])
    res_break_even = break_even_utilization(1.0 - res_hr / od_hr) if od_hr > 0 else 1.0
    if duty >= res_break_even:
        options["reserved"] = {
            "cost": gpu_hours * res_hr, "commit_years": years,
            "reason": f"duty {duty:.0%} clears the {res_break_even:.0%} break-even; "
                      + ("proven steady -> 3yr lock-in" if years == 3
                         else "not steady enough for 3yr -> 1yr commitment"),
        }

    tier = min(options, key=lambda k: options[k]["cost"])
    chosen = options[tier]
    return {
        "tier": tier,
        "commit_years": chosen["commit_years"],
        "cost": chosen["cost"],
        "reason": chosen["reason"],
        "duty": duty,
        "options": {k: round(v["cost"], 2) for k, v in options.items()},
    }
