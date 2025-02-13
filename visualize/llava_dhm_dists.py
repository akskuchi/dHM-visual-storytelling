import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv

plt.style.use('seaborn-v0_8-colorblind')
save_as = 'llava_human_eval_sids.pdf'

def read_scores(fpath):
    if fpath.endswith('.json'):
        return json.load(open(fpath, 'r'))
    with open(fpath, 'r') as fh:
        csvr = csv.reader(fh)
        s = [r for r in csvr][1:]
    fh.close()
    return {v[0]: float(v[1].strip()) for v in s}


def get_dHM_vals(h, m):
    common_h_sids = list(set(h[0].keys()) & set(h[1].keys()) & set(h[2].keys()))
    common_m_sids = list(set(m[0].keys()) & set(m[1].keys()) & set(m[2].keys()))
    common_sids = list(set(common_h_sids) & set(common_m_sids))
    d_HM = {sid: sum([abs(h[idx][sid] - m[idx][sid]) for idx in range(3)]) / 3.0 for sid in common_sids}
    return d_HM


vist_gt_C_G_R = [read_scores(f) for f in ['data/scores/vist/gt_test_C.csv', 
                                          'data/scores/vist/gt_test_G.json', 
                                          'data/scores/vist/gt_test_R.csv']]
llava_C_G_R = [read_scores(f) for f in ['data/scores/vist/llava_vc_p3_C.csv', 
                                        'data/scores/vist/llava_vc_p3_G.json', 
                                        'data/scores/vist/llava_vc_p3_R.csv']]
d_HM_llava_vc = get_dHM_vals(vist_gt_C_G_R, llava_C_G_R)
print(f'd_HM values computed for {len(d_HM_llava_vc)} story-ids')

sids_file = 'data/scores/vist/human-eval/llava_vc_sids.txt'
print(f'story-ids randomly sampled for the human-eval study in file: {sids_file}')

with open(sids_file, 'r') as fh:
    all_sids = fh.readlines()
fh.close()
for_study = [sid.strip() for sid in all_sids]
d_HM_study_b = [d_HM_llava_vc[sid] for sid in for_study]
print(f'd_HM values for {len(d_HM_study_b)} sampled story-ids')

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, dpi=100, constrained_layout=True)
fontsize=24.28
for idx in range(2):
    axs[idx].grid(True, linewidth=0.5, alpha=0.25)
snsplt = sns.histplot(d_HM_study_b, kde=True, color='brown', ax=axs[1], element="step")
snsplt.lines[0].set_color('purple')
snsplt = sns.histplot(d_HM_llava_vc, kde=True, color='brown', ax=axs[0], element="step")
snsplt.lines[0].set_color('purple')
axs[1].set_ylabel('100 samples', fontsize=fontsize)
axs[0].set_xlabel('$d_{HM}$', fontsize=fontsize)
axs[0].set_ylabel('VIST test set', fontsize=fontsize)
axs[1].set_xlabel('$d_{HM}$', fontsize=fontsize)
plt.savefig(save_as, format='pdf') # dpi=1056 # for .png
# plt.show()
print(f'\nsaved plot to: {save_as}')
