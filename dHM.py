import argparse
import csv
import json
import math

import numpy as np


def read_scores_G(fname):
    scores = {}
    s = json.load(open(fname, 'r'))
    for k, v in s.items():
        if v != '' and not math.isnan(float(v)):
            scores[k] = np.tanh(float(v))
    
    return scores


def read_scores_C_R(fname):
    with open(fname, 'r') as fh:
        csvr = csv.reader(fh)
        scores = [r for r in csvr][1:]
    fh.close()

    scores_map = {}
    for r in scores:
        if not math.isnan(float(r[1].strip())):
            scores_map[r[0].strip()] = float(r[1].strip())

    return scores_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dHM evaluator',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, default='VIST',
                        choices=['VIST', 'VWP'], help='dataset to consider')
    args = parser.parse_args()

    gt_split = 'test' # defaulting to test split for all the datasets
    if args.dataset == 'VIST':
        models = ['llava_vc_p3']
    elif args.dataset == 'VWP':
        models = ['llava_vc_p3']
    else:
        raise Exception(f'invalid dataset: {args.dataset}')
    args.dataset = args.dataset.lower()
    
    # 1. read scores for human stories
    h_scores_G = read_scores_G(f'data/scores/{args.dataset}/gt_{gt_split}_G.json')
    h_scores_C = read_scores_C_R(f'data/scores/{args.dataset}/gt_{gt_split}_C.csv')
    h_scores_R = read_scores_C_R(f'data/scores/{args.dataset}/gt_{gt_split}_R.csv')

    # 2. read scores for model-generated stories
    m_scores_G = [read_scores_G(f'data/scores/{args.dataset}/{model}_G.json') for model in models]
    m_scores_C = [read_scores_C_R(f'data/scores/{args.dataset}/{model}_C.csv') for model in models]
    m_scores_R = [read_scores_C_R(f'data/scores/{args.dataset}/{model}_R.csv') for model in models]
    
    # 3. consider sids for which scores are available for all corresponding model-generated and human stories
    common_sids = {}
    for model_scores in [m_scores_G, m_scores_C, m_scores_R]:
        for s in model_scores:
            for sid in s.keys():
                common_sids[sid] = common_sids[sid] + 1 if sid in common_sids else 1
    for s in [h_scores_G, h_scores_C, h_scores_R]:
        for sid in s.keys():
            common_sids[sid] = common_sids[sid] + 1 if sid in common_sids else 1

    # 4. for each COMMON sid compute dG, dC, dR, and dHM
    dG, dC, dR, dHM = {}, {}, {}, {}
    for sid, cnt in common_sids.items():
        if cnt == (len(models) + 1) * 3:
            dG[sid] = [abs(h_scores_G[sid] - model_scores[sid]) for model_scores in m_scores_G]
            dC[sid] = [abs(h_scores_C[sid] - model_scores[sid]) for model_scores in m_scores_C]
            dR[sid] = [abs(h_scores_R[sid] - model_scores[sid]) for model_scores in m_scores_R]
            dHM[sid] = [(dG[sid][idx] + dC[sid][idx] + dR[sid][idx]) / 3 for idx in range(len(models))]
            
    # 5. obtain overall aggregate distances per model by computing average
    model_distances = {}
    for idx, model in enumerate(models):
        model_distances[model] = {
            'dG': round(np.mean([v[idx] for _, v in dG.items()]), 4),
            'dC': round(np.mean([v[idx] for _, v in dC.items()]), 4),
            'dR': round(np.mean([v[idx] for _, v in dR.items()]), 4),
            'dHM': round(np.mean([v[idx] for _, v in dHM.items()]), 4)
        }


    print(json.dumps(model_distances, indent=4))
