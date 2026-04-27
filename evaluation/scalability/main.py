
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

EVALS = Path(r'C:\workspace\ml\workspace\master\original\evals')

# ── Load FLOPs ────────────────────────────────────────────────────────────────
with open(EVALS / 'flops_breakdown_results_originial.json') as f:
    flops_orig = json.load(f)
with open(EVALS / 'flops_breakdown_results_sliced.json') as f:
    flops_sliced = json.load(f)

def parse_flops(flops_data, heads_wanted):
    seen, entries = set(), []
    for e in flops_data['detr_resnet50']:
        h = e['heads']
        if h not in seen:
            seen.add(h)
            entries.append(e)
    entries.sort(key=lambda x: x['heads'])
    MEAN = 0
    return {e['heads']: e['detr_flops'][MEAN] for e in entries if e['heads'] in heads_wanted}

# ── Load precision ────────────────────────────────────────────────────────────
HEADS = [1, 2, 3, 4]

baseline_ap = {}
for h in HEADS:
    with open(EVALS / f'baseline_test_stats_heads{h}.json') as f:
        d = json.load(f)
    baseline_ap[h] = d['coco_eval_bbox'][0]

with open(EVALS / 'sliced_model4_head_detr_stats.json') as f:
    sliced_data = json.load(f)
sliced_ap = {h: sliced_data[f'heads_{h}']['coco_eval_bbox'][0] for h in HEADS}

orig_gmac  = parse_flops(flops_orig,   set(HEADS))
sliced_gmac = parse_flops(flops_sliced, set(HEADS))

orig_x  = [orig_gmac[h]   for h in HEADS]
sliced_x = [sliced_gmac[h] for h in HEADS]
orig_y  = [baseline_ap[h]  for h in HEADS]
sliced_y = [sliced_ap[h]   for h in HEADS]

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-poster')
C_BASE   = '#1B4F72'   # dark blue  – baseline
C_SLICE  = '#C0392B'   # dark red   – sliced

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(orig_x,   orig_y,   color=C_BASE,  linewidth=2.5,
        marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2, zorder=3)
ax.plot(sliced_x, sliced_y, color=C_SLICE, linewidth=2.5,
        marker='s', markersize=9, markerfacecolor='white', markeredgewidth=2, zorder=3)

for x, y, h in zip(orig_x, orig_y, HEADS):
    ax.annotate(f'{h}h', xy=(x, y), xytext=(6, 4),
                textcoords='offset points', fontsize=10, color=C_BASE, fontweight='bold')
for x, y, h in zip(sliced_x, sliced_y, HEADS):
    ax.annotate(f'{h}h', xy=(x, y), xytext=(6, -12),
                textcoords='offset points', fontsize=10, color=C_SLICE, fontweight='bold')

# Delta annotations between matched head counts
for h, xo, xs, yo, ys in zip(HEADS, orig_x, sliced_x, orig_y, sliced_y):
    mid_x = (xo + xs) / 2
    mid_y = (yo + ys) / 2
    delta_ap   = ys - yo
    delta_gmac = xs - xo
    ax.annotate(
        f'ΔAP={delta_ap:+.3f}\nΔGMAC={delta_gmac:+.2f}',
        xy=(mid_x, mid_y), xytext=(12, 0), textcoords='offset points',
        fontsize=7.5, color='#555555',
        arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.8)
    )

legend_handles = [
    mlines.Line2D([], [], color=C_BASE,  marker='o', linewidth=2.5, markersize=8,
                  markerfacecolor='white', label='Baseline'),
    mlines.Line2D([], [], color=C_SLICE, marker='s', linewidth=2.5, markersize=8,
                  markerfacecolor='white', label='Sliced (4-head model)'),
]
ax.legend(handles=legend_handles, fontsize=11, frameon=True)

ax.set_xlabel('Total GMACs (mean)', fontsize=13, fontweight='bold')
ax.set_ylabel('AP@[.50:.95]',       fontsize=13, fontweight='bold')
ax.set_title('Baseline vs Sliced DETR-ResNet50: GMACs vs AP@[.50:.95]',
             fontsize=13, fontweight='bold', pad=15)

ax.grid(True, alpha=0.4, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(EVALS.parent / 'working' / 'jupyter' / 'baseline_vs_sliced_gmac_precision.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Heads':>5} | {'Baseline':>14} | {'Sliced':>14} | {'ΔAP':>8} | {'Baseline GMACs':>15} | {'Sliced GMACs':>13} | {'ΔGMAC':>7}")
print("-" * 90)
for h in HEADS:
    bap  = baseline_ap[h]
    sap  = sliced_ap[h]
    bgm  = orig_gmac[h]
    sgm  = sliced_gmac[h]
    print(f"{h:>5} | {bap:>14.4f} | {sap:>14.4f} | {sap-bap:>+8.4f} | {bgm:>15.2f} | {sgm:>13.2f} | {sgm-bgm:>+7.4f}")
