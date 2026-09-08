# ============================================================================
#  A(g): reliability weighting as a function of the intensity gap
#  ---------------------------------------------------------------------------
#  WHERE TO PUT THIS:  append as a NEW CELL at the END of
#                      09_masked_sweep_v8.ipynb
#
#  FIXED (v2): nb09's load() keeps only {X, y, types} and DROPS aud_int/vis_int,
#              which caused  KeyError: 'aud_int'.  The intensities are now read
#              straight from the test .npz instead of from the `test` dict.
#
#  RUNTIME: ~25-45 min on CPU for 30 networks (3 sizes x 5 seeds x 2 conflicts).
#           Set SIZES = [2] first for a ~2 min smoke test.
#           Per-trial predictions are cached, so re-running is instant.
# ============================================================================

import numpy as np, torch, matplotlib.pyplot as plt

SIZES     = [2, 3, 5]          # sizes that carry the argument in the report
KIND      = "standard"         # "standard" or "masked"
N_BINS    = 5
CACHE     = OUT_DIR / f"Ag_conflict_v8_{KIND}.npz"
CONFLICTS = ["loc_conflict_audL_visR", "loc_conflict_audR_visL"]

# ------------------------------------------------- intensities (THE FIX)
# nb09's load() drops aud_int/vis_int, so read them from the npz directly.
_raw     = np.load(DATA_DIR / "test.npz", allow_pickle=True)
_aud     = _raw["aud_int"]
_vis     = _raw["vis_int"]

conf_mask = np.isin(test["types"], CONFLICTS)
gap       = np.abs(_aud - _vis)[conf_mask]     # |aud - vis|, the reliability gap
y_conf    = test["y"][conf_mask]               # already the "stronger cue wins" label
print(f"{conf_mask.sum()} conflict trials, gap range {gap.min():.2f} to {gap.max():.2f}")

# ------------------------------------------------------------ train / load
if CACHE.exists():
    preds = np.load(CACHE, allow_pickle=True)["preds"].item()
    print(f"Loaded cached predictions from {CACHE.name} (no retraining).")
else:
    preds = {}
    for h in SIZES:
        for s in SEEDS:
            model = train_model(KIND, h, s)
            model.eval()
            with torch.no_grad():
                p = model(torch.from_numpy(test["X"]))[:, -1, :].argmax(-1).numpy()
            preds[(h, s)] = p[conf_mask]
            print(f"  h={h} seed={s}  pooled agreement = "
                  f"{float((preds[(h,s)] == y_conf).mean()):.3f}")
    np.savez(CACHE, preds=np.array(preds, dtype=object))
    print(f"Saved predictions to {CACHE.name}")

# --------------------------------------------------- bin by intensity gap
edges       = np.quantile(gap, np.linspace(0, 1, N_BINS + 1))   # equal-count bins
edges[-1]  += 1e-9
centres     = 0.5 * (edges[:-1] + edges[1:])
bin_id      = np.clip(np.digitize(gap, edges) - 1, 0, N_BINS - 1)

curves = {}
for h in SIZES:
    curves[h] = np.array([
        [ (preds[(h, s)] == y_conf)[bin_id == b].mean() if (bin_id == b).any() else np.nan
          for b in range(N_BINS) ]
        for s in SEEDS
    ])

# ------------------------------------------------------------------ report
print(f"\nA(g): agreement with the stronger cue, mean +/- std across "
      f"{len(SEEDS)} seeds  ({KIND} GRU)")
print("  gap bin   " + "".join(f"{c:>15.2f}" for c in centres))
print("  " + "-" * (10 + 15 * N_BINS))
for h in SIZES:
    m, sd = np.nanmean(curves[h], 0), np.nanstd(curves[h], 0)
    print(f"  {h} units  " + "".join(f"   {a:.2f} +/- {b:.2f}" for a, b in zip(m, sd)))

# -------------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(7.0, 4.6))
colours = {2: "#7ab3e0", 3: "#2e6ca4", 5: "#14405f"}
for h in SIZES:
    m, sd = np.nanmean(curves[h], 0), np.nanstd(curves[h], 0)
    c = colours.get(h)
    ax.plot(centres, m, marker="o", lw=2.0, ms=6, color=c, label=f"{h} hidden units")
    ax.errorbar(centres, m, yerr=sd, fmt="none", capsize=3, lw=1.1, color=c)
    ax.fill_between(centres, m - sd, m + sd, alpha=0.13, color=c)
ax.axhline(0.5, ls="--", color="0.5", lw=1.2)
ax.text(centres[0], 0.515, "chance (0.5)", fontsize=8, color="0.45")
ax.set_xlabel("intensity gap  |aud - vis|")
ax.set_ylabel("agreement with the stronger cue")
ax.set_title(f"Reliability weighting: A(g) across seeds ({KIND} GRU)", fontsize=11)
ax.set_ylim(0.3, 1.02); ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / f"Ag_reliability_curve_{KIND}.png", dpi=180, bbox_inches="tight")
plt.show()
print("\nSaved figure to", OUT_DIR / f"Ag_reliability_curve_{KIND}.png")

# ------------------------------------------------------------ sanity check
print("\nPooled agreement over ALL conflict trials (the conservative number):")
for h in SIZES:
    print(f"  h={h}: {np.mean([(preds[(h,s)] == y_conf).mean() for s in SEEDS]):.3f}")
print("A rising A(g) alongside a modest pooled value is the signature of")
print("reliability weighting; a fixed-side policy would stay flat near 0.5.")
