import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
 e\ml\workspace\master\original\evals')
MEAN = 0

def parse_flops_series(data, model_key):
    seen, entries = set(), []
    for e in data[model_key]:
        h = e['heads']
        if h not in seen:
            seen.add(h)
            entries.append(e)
    entries.sort(key=lambda x: x['heads'])
    return [e['heads'] for e in entries], [e['detr_flops'][MEAN] for e in entries]

with open(EVALS / 'flops_breakdown_results_originial.json') as f:
    orig_data = json.load(f)
with open(EVALS / 'flops_breakdown_results_sliced.json') as f:
    sliced_data = json.load(f)
with open(EVALS / 'flops_breakdown_results_local_head.json') as f:
    local_data = json.load(f)

orig_heads, orig_flops = parse_flops_series(orig_data, 'detr_resnet50')
sliced_heads, sliced_flops = parse_flops_series(sliced_data, 'sliced_detr_resnet50')
local_heads, local_flops = parse_flops_series(local_data, 'local_head_detr_resnet50')

C_ORIG = '#1B4F72'
C_SLICED = '#C0392B'
C_LOCAL = '#1E8449'

plt.style.use('seaborn-v0_8-poster')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(orig_heads, orig_flops, color=C_ORIG, linewidth=2.5, linestyle='-.',
        marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2, zorder=4)
ax.plot(sliced_heads, sliced_flops, color=C_SLICED, linewidth=2.5, linestyle='--',
        marker='s', markersize=9, markerfacecolor='white', markeredgewidth=2, zorder=3)
ax.plot(local_heads, local_flops, color=C_LOCAL, linewidth=2.5, linestyle='-.',
        marker='^', markersize=9, markerfacecolor='white', markeredgewidth=2, zorder=3)

legend_handles = [
    mlines.Line2D([], [], color=C_ORIG,   linestyle='-',  marker='o', linewidth=2.5, markersize=8,
                  markerfacecolor='white', label='Original'),
    mlines.Line2D([], [], color=C_SLICED, linestyle='--', marker='s', linewidth=2.5, markersize=8,
                  markerfacecolor='white', label='Sliced'),
    mlines.Line2D([], [], color=C_LOCAL,  linestyle='-.', marker='^', linewidth=2.5, markersize=8,
                  markerfacecolor='white', label='Local-Head'),
]
ax.legend(handles=legend_handles, fontsize=11, frameon=True)

ax.set_xlabel('Number of heads', fontsize=13, fontweight='bold')
ax.set_ylabel('Total GMACs (mean)', fontsize=13, fontweight='bold')
ax.set_title('DETR-ResNet50: Total GMACs vs Number of Heads', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(sorted(set(orig_heads + sliced_heads + local_heads)))
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(EVALS.parent / 'working' / 'jupyter' / 'detr_flops_per_head.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'Heads':>5} | {'Original':>12} | {'Sliced':>12} | {'Local-Head':>12} | {'Δ Sliced':>10} | {'Δ Local':>10}")
print("-" * 70)
orig_map = dict(zip(orig_heads, orig_flops))
sliced_map = dict(zip(sliced_heads, sliced_flops))
local_map = dict(zip(local_heads, local_flops))
for h in sorted(set(orig_heads) | set(sliced_heads) | set(local_heads)):
    o = orig_map.get(h)
    s = sliced_map.get(h)
    l = local_map.get(h)
    ds = f'{s - o:+.4f}' if o is not None and s is not None else '-'
    dl = f'{l - o:+.4f}' if o is not None and l is not None else '-'
    o_str = f'{o:12.4f}' if o is not None else f"{'−':>12}"
    s_str = f'{s:12.4f}' if s is not None else f"{'−':>12}"
    l_str = f'{l:12.4f}' if l is not None else f"{'−':>12}"
    print(f"{h:>5} | {o_str} | {s_str} | {l_str} | {ds:>10} | {dl:>10}")
