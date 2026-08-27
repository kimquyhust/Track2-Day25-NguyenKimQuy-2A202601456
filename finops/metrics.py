"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


# --- Your Turn #2: right-sizing memory-bound GPUs ----------------------------
def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """$/hour per GB of HBM — the unit price that matters for KV-cache capacity."""
    return on_demand_hr / hbm_gb if hbm_gb > 0 else float("inf")


def dollars_per_tbs(on_demand_hr: float, peak_bw_tbs: float) -> float:
    """$/hour per TB/s of HBM bandwidth — the unit price of decode throughput."""
    return on_demand_hr / peak_bw_tbs if peak_bw_tbs > 0 else float("inf")


def is_memory_bound(mfu: float, mbu: float, margin: float = 1.05) -> bool:
    """A GPU whose bandwidth utilisation outruns its FLOPs utilisation is decode-bound.

    Buying more FLOPs for such a GPU changes nothing: the bottleneck is moving
    weights and KV-cache through HBM, not the tensor cores.
    """
    return mbu > mfu * margin


def rightsize_candidate(current_type: str, catalog: dict, achieved_bw_tbs: float,
                        vram_needed_gb: float, headroom: float = 1.20) -> dict | None:
    """Cheapest GPU that still covers the observed bandwidth and memory footprint.

    ``catalog`` maps gpu_type -> dict with ``on_demand_hr``, ``hbm_gb`` and
    ``peak_bw_tbs``.  Returns None when the current GPU is already the cheapest
    box that fits — the honest answer for a well-sized workload.
    """
    current = catalog.get(current_type)
    if not current:
        return None
    cur_hr = float(current["on_demand_hr"])
    need_bw = achieved_bw_tbs * headroom
    need_gb = vram_needed_gb * headroom

    best = None
    for gtype, row in catalog.items():
        hr = float(row["on_demand_hr"])
        if hr >= cur_hr:
            continue
        if float(row["peak_bw_tbs"]) < need_bw or float(row["hbm_gb"]) < need_gb:
            continue
        if best is None or hr < float(catalog[best]["on_demand_hr"]):
            best = gtype
    if best is None:
        return None
    new_hr = float(catalog[best]["on_demand_hr"])
    return {
        "from": current_type,
        "to": best,
        "from_hr": cur_hr,
        "to_hr": new_hr,
        "hourly_savings": cur_hr - new_hr,
        "savings_pct": (1.0 - new_hr / cur_hr) * 100.0,
        "from_usd_per_gb": dollars_per_gb_vram(cur_hr, float(current["hbm_gb"])),
        "to_usd_per_gb": dollars_per_gb_vram(new_hr, float(catalog[best]["hbm_gb"])),
        "from_usd_per_tbs": dollars_per_tbs(cur_hr, float(current["peak_bw_tbs"])),
        "to_usd_per_tbs": dollars_per_tbs(new_hr, float(catalog[best]["peak_bw_tbs"])),
    }
