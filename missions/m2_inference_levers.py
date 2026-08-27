"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Also implements Your Turn #3 (cache economics measured from real traffic) and
Your Turn #4 (reasoning budget + a routing rule with a costed what-if).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
CACHE_READ_DISCOUNT = 0.10
ACTIVE_CACHE_TTL = "5-min"          # policy we actually buy
# Your Turn #4 routing rule: reasoning is reserved for genuinely long prompts.
REASONING_MIN_INPUT_TOKENS = 3000
REASONING_TRAFFIC_CAPS = (0.10, 0.05)


def observed_cache_reads(rows: list[dict]) -> dict:
    """Average re-reads per cached prompt, measured from token_usage.csv.

    Every cached input token was written once and re-read afterwards, so
    ``cached / (input - cached)`` is the number of times an average cache entry
    is read back per fresh write.  This replaces the usual hand-waved guess.
    """
    acc: dict[str, list[float]] = {}
    for r in rows:
        tier = r["route_tier"]
        inp = num(r["input_tokens"])
        cached = min(num(r["cached_input_tokens"]), inp)
        a = acc.setdefault(tier, [0.0, 0.0])
        a[0] += cached
        a[1] += inp - cached
    return {tier: (cached / fresh if fresh > 0 else 0.0) for tier, (cached, fresh) in acc.items()}


def cache_policy_table(reads_by_tier: dict) -> list[dict]:
    """Break-even vs. observed reuse for every (TTL policy x model tier) pair."""
    out = []
    for ttl, write_cost in pricing.CACHE_TTL_POLICIES.items():
        be = pricing.cache_break_even_reads(write_cost, CACHE_READ_DISCOUNT)
        for tier, (price_in, _) in MODEL_PRICES.items():
            reads = reads_by_tier.get(tier, 0.0)
            out.append({
                "ttl": ttl, "tier": tier, "write_cost": write_cost,
                "break_even_reads": be, "observed_reads": reads,
                "worth_it": pricing.cache_is_worth_it(reads, write_cost, CACHE_READ_DISCOUNT),
                "usd_per_m_saved": pricing.cache_savings_per_million(
                    price_in, reads, write_cost, CACHE_READ_DISCOUNT),
            })
    return out


def reasoning_scenarios(rows: list[dict], opt_prices: dict) -> dict:
    """Cost/energy of reasoning traffic today, and under caps + a routing rule.

    Downgrading a reasoning request does not just switch a flag: the model stops
    emitting its long internal chain, so we re-price it at the *observed* mean
    output length of ordinary traffic and drop the 80x energy multiplier.
    """
    reasoning, normal = [], []
    for r in rows:
        (reasoning if int(num(r["is_reasoning"])) else normal).append(r)
    normal_out_mean = (sum(num(r["output_tokens"]) for r in normal) / len(normal)) if normal else 0.0

    def price(r, output_tokens, is_reasoning):
        pin, pout = opt_prices[r["route_tier"]]
        cost = pricing.request_cost(
            int(num(r["input_tokens"])), int(output_tokens), pin, pout,
            cached_in=int(num(r["cached_input_tokens"])),
            batch=bool(int(num(r["is_batch"]))))
        wh = sustainability.wh_per_query(
            int(num(r["input_tokens"])) + int(output_tokens), is_reasoning=is_reasoning)
        return cost, wh

    base_cost = base_wh = 0.0
    for r in reasoning:
        c, w = price(r, num(r["output_tokens"]), True)
        base_cost += c
        base_wh += w
    normal_cost = normal_wh = 0.0
    for r in normal:
        c, w = price(r, num(r["output_tokens"]), False)
        normal_cost += c
        normal_wh += w

    # Most complex prompts keep reasoning; the rest get downgraded.
    ranked = sorted(reasoning, key=lambda r: num(r["input_tokens"]), reverse=True)

    def simulate(keep: list[dict], label: str) -> dict:
        keep_ids = {id(r) for r in keep}
        cost = wh = 0.0
        for r in reasoning:
            if id(r) in keep_ids:
                c, w = price(r, num(r["output_tokens"]), True)
            else:
                c, w = price(r, normal_out_mean, False)
            cost += c
            wh += w
        return {
            "label": label,
            "kept": len(keep),
            "traffic_pct": len(keep) / len(rows) * 100 if rows else 0.0,
            "reasoning_cost": cost, "reasoning_wh": wh,
            "cost_saved": base_cost - cost, "wh_saved": base_wh - wh,
            "total_cost": cost + normal_cost, "total_wh": wh + normal_wh,
        }

    scenarios = []
    for cap in REASONING_TRAFFIC_CAPS:
        budget = int(len(rows) * cap)
        scenarios.append(simulate(ranked[:budget], f"Đặt trần {cap:.0%} traffic"))
    rule_keep = [r for r in reasoning if num(r["input_tokens"]) >= REASONING_MIN_INPUT_TOKENS]
    scenarios.append(simulate(rule_keep, f"Quy tắc routing: prompt ≥ {REASONING_MIN_INPUT_TOKENS:,} token"))

    return {
        "requests": len(reasoning),
        "traffic_pct": len(reasoning) / len(rows) * 100 if rows else 0.0,
        "cost": base_cost, "wh": base_wh,
        "normal_cost": normal_cost, "normal_wh": normal_wh,
        "cost_pct": base_cost / (base_cost + normal_cost) * 100 if (base_cost + normal_cost) else 0.0,
        "wh_pct": base_wh / (base_wh + normal_wh) * 100 if (base_wh + normal_wh) else 0.0,
        "mean_output_reasoning": (sum(num(r["output_tokens"]) for r in reasoning) / len(reasoning)) if reasoning else 0.0,
        "mean_output_normal": normal_out_mean,
        "energy_multiplier": sustainability.REASONING_ENERGY_MULTIPLIER,
        "scenarios": scenarios,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    reads_by_tier = observed_cache_reads(rows)
    policies = cache_policy_table(reads_by_tier)
    write_cost = pricing.CACHE_TTL_POLICIES[ACTIVE_CACHE_TTL]
    # Cache is enabled per model tier, and only where the measured reuse clears
    # that tier's break-even — no blanket "caching is always good" assumption.
    cache_on = {tier: pricing.cache_is_worth_it(reads, write_cost, CACHE_READ_DISCOUNT)
                for tier, reads in reads_by_tier.items()}

    base_cost = opt_cost = no_cache_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        eligible_cached = cached if cache_on.get(r["route_tier"], False) else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=eligible_cached, batch=is_batch)
        no_cache_cost += pricing.request_cost(inp, out, pin, pout, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    reasoning = reasoning_scenarios(rows, MODEL_PRICES)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- cache economics (measured, not assumed) --")
        print(f"{'TTL':8}{'tier':7}{'break-even':>12}{'observed':>10}{'verdict':>10}{'$/1M saved':>12}")
        for p in policies:
            print(f"{p['ttl']:8}{p['tier']:7}{p['break_even_reads']:>12.2f}{p['observed_reads']:>10.2f}"
                  f"{('WORTH IT' if p['worth_it'] else 'NO'):>10}{p['usd_per_m_saved']:>12.4f}")
        print(f"active policy = {ACTIVE_CACHE_TTL} cache; enabled per tier: {cache_on}")
        print(f"cache contributes ${no_cache_cost - opt_cost:,.2f}/day of the optimized bill")

        print("\n-- reasoning budget --")
        print(f"today: {reasoning['requests']}/{len(rows)} requests ({reasoning['traffic_pct']:.1f}% of traffic) "
              f"-> ${reasoning['cost']:,.2f}/day ({reasoning['cost_pct']:.1f}% of cost), "
              f"{reasoning['wh']:,.0f} Wh/day ({reasoning['wh_pct']:.1f}% of energy)")
        print(f"mean output: reasoning {reasoning['mean_output_reasoning']:,.0f} tok vs normal "
              f"{reasoning['mean_output_normal']:,.0f} tok "
              f"({reasoning['mean_output_reasoning']/reasoning['mean_output_normal']:.1f}x longer)")
        for s in reasoning["scenarios"]:
            print(f"  {s['label']:42} keep {s['kept']:>4} ({s['traffic_pct']:.1f}% traffic)  "
                  f"save ${s['cost_saved']:,.2f}/day, {s['wh_saved']:,.0f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_reads_by_tier": {k: round(v, 3) for k, v in reads_by_tier.items()},
        "cache_policies": policies,
        "cache_enabled_by_tier": cache_on,
        "cache_savings_daily": round(no_cache_cost - opt_cost, 2),
        "active_cache_ttl": ACTIVE_CACHE_TTL,
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
