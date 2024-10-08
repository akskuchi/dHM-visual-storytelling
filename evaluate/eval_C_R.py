import argparse
import json

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn.functional import softmax

from rovist_c import SOPClassifier, model_nm, tokenizer

nltk.download('punkt')
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'using device: {device}\n')


def get_coherence_score(story, model, debug=False):
    sentences = nltk.sent_tokenize(story)
    prefix = sentences[0]
    scores = []
    for pos in range(1, len(sentences)):
        sentence = sentences[pos]

        if prefix.strip() == sentence.strip():
            scores.append(0)
            continue

        prefix_len = len(tokenizer.tokenize(prefix))
        sentence_len = len(tokenizer.tokenize(sentence))

        if debug:
            print(f'prefix: {prefix}, sentence: {sentence}')
        input_text = ['[CLS]', prefix, '[SEP]', sentence, '[SEP]']
        input_text = ' '.join(input_text)
        input_tokens = tokenizer.tokenize(input_text)
        
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids = torch.tensor([input_ids], device=device)

        attention_ids = (prefix_len + 2) * [1] + (sentence_len + 1) * [1]
        attention_ids = torch.tensor([attention_ids], device=device)

        segment_ids = (prefix_len + 2) * [0] + (sentence_len + 1) * [1]
        segment_ids = torch.tensor([segment_ids], device=device)

        with torch.no_grad():
            sop_logits = model(input_ids, attention_ids, segment_ids)

        sop_probs = softmax(sop_logits, dim=1)
        scores.append(sop_probs[0][1].item())
        prefix += ' ' + sentence # considers prefix, CHANGE made to the original RoViST-C metric

    if debug:
        print(f'scores: {scores}')
    return scores


def eval_coherence(sids, stories, debug):
    print('evaluating coherence:\nloading pre-trained ALBERT model...', end='')
    checkpoint = torch.load('data/sop_model_epoch4.pth.tar', map_location=device)
    checkpoint_opts = checkpoint['opt']
    model = SOPClassifier(checkpoint_opts.hidden_dim, 
                          checkpoint_opts.dropout_prob,
                          model_nm)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print('complete.\n')
    
    c_scores = []
    valid_sids, invalid_sids = [], []
    for sid in tqdm(sids, disable=debug):
        try:
            scores = get_coherence_score(stories[sid], model, debug)
            if len(scores) == 0:
                invalid_sids.append(sid)
            else:
                c_scores.append(np.mean(scores))
                valid_sids.append(sid)
        except Exception as e:
            if debug:
                print(f'cannot evaluate {sid}, exception: {e}')
            invalid_sids.append(sid)

    if len(invalid_sids) > 0:
        print(f'coherence scores not computed for {len(invalid_sids)} sequences:\n{invalid_sids}\n')
    
    print('\n')
    return valid_sids, c_scores


def eval_repetition(sids, stories, debug):
    print('evaluating repetition:')
    r_scores = []
    valid_sids, invalid_sids = [], []
    for sid in tqdm(sids, disable=debug):
        story = stories[sid]
        sentences = nltk.sent_tokenize(story)
        sentence_words = []
        for sentence in sentences:
            sentence_words.append(nltk.word_tokenize(sentence)[:-1])

        inter_sentence_scores = []
        for i in range(1, len(sentence_words)):
            next_ngrams = sentence_words[i]
            for j in range(0, i):
                prev_ngrams = sentence_words[j]
                union = len(set(prev_ngrams + next_ngrams))
                if union == 0:
                    continue
                intersection = 0
                for ngram_i in next_ngrams:
                    for ngram_j in prev_ngrams:
                        intersection += int(ngram_i == ngram_j)
                
                inter_sentence_scores.append(intersection / union)

        intra_sentence_scores = []
        ngram_len = 4
        for _sentence_words in sentence_words:
            for i in range(0, len(_sentence_words), ngram_len):
                j = i + ngram_len
                prev_slice = _sentence_words[i : i + ngram_len]
                next_slice = _sentence_words[j : j + ngram_len]
                if len(next_slice) == 0:
                    continue
                union = len(set(prev_slice + next_slice))
                if union == 0:
                    continue
                intersection = 0
                for word_i in prev_slice:
                    for word_j in next_slice:
                        intersection += int(word_i == word_j)
                
                intra_sentence_scores.append(intersection / union)
        
        r_score = 1 - np.mean([np.mean(inter_sentence_scores), 
                                np.mean(intra_sentence_scores) if len(intra_sentence_scores) > 0 else 0])
        if np.isnan(r_score):
            invalid_sids.append(sid)
            if debug:
                print(f'cannot evaluate {sid}, NaN scores')
        else:
            valid_sids.append(sid)
            r_scores.append(r_score)

    if len(invalid_sids) > 0:
        print(f'repetition scores not computed for {len(invalid_sids)} sequences:\n{invalid_sids}\n')
        
    print('\n')
    return valid_sids, r_scores


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='coherence and repetition evaluator',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input_file', type=str, 
                        default='data/stories/vist/gt_test.json',
                        help='path to file with stories (?.json)')
    parser.add_argument('-o', '--output_file', type=str,
                        default='data/scores/vist/gt_test', 
                        help='scores output file (?)')
    parser.add_argument('-m', '--mode', type=str, 
                        default='C_R', 
                        choices=['C_R', 'C', 'R'],
                        help='metric to compute, default: C_R')
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='pass --debug to see verbose logs')
    parser.add_argument('-s', '--sample_run',
                        action='store_true',
                        help='pass --sample_run to compute score(s) for 5 sample stories')
    args = parser.parse_args()

    stories = json.load(open(args.input_file))
    print(f'read stories from {args.input_file}')

    sids_to_eval = list(stories.keys())[:(5 if args.sample_run else len(stories.keys()))]
    print(f'{len(sids_to_eval)} samples to score')

    scores_C, scores_R = [], []
    sids_C, sids_R = [], []
    if args.mode == 'C_R':
        sids_C, scores_C = eval_coherence(sids_to_eval, stories, args.debug)
        sids_R, scores_R = eval_repetition(sids_to_eval, stories, args.debug)
    elif args.mode == 'C':
        sids_C, scores_C = eval_coherence(sids_to_eval, stories, args.debug)
    elif args.mode == 'R':
        sids_R, scores_R = eval_repetition(sids_to_eval, stories, args.debug)

    
    for mode in ['C', 'R']:
        metric, scores, sids = ('coherence', scores_C, sids_C) if mode == 'C' else ('repetition', scores_R, sids_R)
        if len(sids) > 0:
            save_as = f'{args.output_file}_{mode}.csv'
            df = pd.DataFrame()
            df['story_id'] = sids
            df[f'{mode}_score'] = scores
            df.to_csv(f'{save_as}', index=False)
            print(f'saved {metric} scores to {save_as}')
            print(f'overall {metric} score: {df[f"{mode}_score"].mean()}\n')
