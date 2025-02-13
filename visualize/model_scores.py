import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-colorblind')

model_names = tuple(reversed(['AREL', 'GLAC Net', 'TAPM', 'BLIP-2', 'LLaVA']))
legend_box, xlim, xtick_lim, bar_label_pad = (0.710, 0.50), 0.81, 0.81, (0.059, -0.10)
save_as = 'sec4_dhm.pdf'

data = {
    'dHM': {'AREL': 0.24883975356786933, 'GLAC Net': 0.21644969758081228, 'TAPM': 0.2135605804935911, 'BLIP-2': 0.34543997577556035, 'LLaVA': 0.16003820673716412}, 
    'dG': {'AREL': 0.29573068879818326, 'GLAC Net': 0.3707947679589893, 'TAPM': 0.2374075119762508, 'BLIP-2': 0.2154685810352121, 'LLaVA': 0.2074355583024042}, 
    'dC': {'AREL': 0.34078817808093115, 'GLAC Net': 0.24021097612717537, 'TAPM': 0.2890091291015458, 'BLIP-2': 0.5535012966858701, 'LLaVA': 0.14757019087255296}, 
    'dR': {'AREL': 0.11000039382449345, 'GLAC Net': 0.038343348656272216, 'TAPM': 0.11426510040297677, 'BLIP-2': 0.2673500496055989, 'LLaVA': 0.1251088710365352}
}

metric_level_dhm_scaled = {
    'c': [data['dC'][model] / 3.0 for model in model_names],
    'g': [data['dG'][model] / 3.0 for model in model_names],
    'r': [data['dR'][model] / 3.0 for model in model_names]
}
overall_dhm = [data['dHM'][model] for model in model_names]

scores = {
    "Coherence": metric_level_dhm_scaled['c'],
    "Visual grounding": metric_level_dhm_scaled['g'],
    "Repetition": metric_level_dhm_scaled['r'],
}

fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100, constrained_layout=True)
ax.grid(True, linewidth=0.5, alpha=0.25)
fontsize=21

colors, hatches = ['xkcd:heather', 'xkcd:light mustard', 'xkcd:pastel blue'], ['|', 'x', '+']
width, idx = 0.40, 0
left = np.zeros(len(model_names))
for model_name, score in scores.items():
    p = ax.barh(model_names, score, width, label=model_name, left=left, color=colors[idx], hatch=hatches[idx], alpha=.99)
    left += score
    idx += 1
    
y_offset = 4
for i, d in enumerate(overall_dhm):
    ax.text(d + bar_label_pad[0], i + bar_label_pad[1], round(d, 3), ha='center', fontsize=fontsize, fontweight='normal')

plt.legend(loc="center", fontsize=fontsize, ncol=1, bbox_to_anchor=legend_box, fancybox=True, frameon=True, framealpha=1)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
ax.set_xticks(np.arange(0.00, xtick_lim, 0.1))
ax.set_xlim(0, xlim)
ax.set_xlabel('distance $d_{HM}$', fontsize=fontsize, fontweight=550)
plt.savefig(save_as, format='pdf') # dpi=1056 # for .png
# plt.show()
print(f'saved plot to: {save_as}')
