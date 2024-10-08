import os
import json
import argparse

from tqdm import tqdm
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}\n')

num_to_word = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten'
}

prompts_VC = {
    'P1': '[INST] <image>\nWrite a story using exactly [#SENTS] sentences for this image sequence. Do not use more than [#SENTS] sentences. [/INST]',
    'P2': '[INST] <image>\nGenerate a story consisting of [#SENTS] sentences for this image sequence. Use only [#SENTS] sentences and not more. [/INST]',
    'P3': '[INST] <image>\nOutput a story about this sequence of images using only [#SENTS] sentences. Make sure the story does not include more than [#SENTS] sentences. [/INST]'
}

prompts_LC = {
    'P1': '[INST] <image>\nUsing this image, add one sentence to the following story: "{}<s>" [/INST]',
    'P2': '[INST] <image>\nGiven this image, write one sentence as an addition to the following story: "{}<s>" [/INST]',
    'P3': '[INST] <image>\nAdd one sentence based on this image to the following story: "{}<s>" [/INST]'
}


def load_model():
    version = 'llava-hf/llava-v1.6-mistral-7b-hf'
    processor = LlavaNextProcessor.from_pretrained(version)
    model = LlavaNextForConditionalGeneration.from_pretrained(version, torch_dtype=torch.float16)
    model.to(device)
    return model, processor


def load_data(dataset, data_path, sample_run=True):
    if dataset == 'VIST':
        data = {}
        imgs, txts = [], []
        annos = json.load(open(f'{data_path}/test.story-in-sequence.json'))['annotations']
        for idx in range(len(annos)):
            imgs.append(f'{data_path}/images/{annos[idx][0]["photo_flickr_id"]}')
            txts.append(annos[idx][0]['text'])
            if len(imgs) == 5:
                data[annos[idx][0]['story_id']] = (imgs, txts)
                imgs, txts = [], []

        return {'49620': data['49620']} if sample_run else data
    elif dataset == 'VWP':
        sceneid_2_imgs = json.load(open(f'{data_path}/sceneid_2_imgs.json'))
        test_set = json.load(open(f'{data_path}/split_info.json'))['test']
        test_stories = json.load(open(f'{data_path}/test.json'))
        
        data = {sid: (sceneid_2_imgs[sid], test_stories[sid]) for sid in test_set}

        return {'tt0104348_0007_6;2': data['tt0104348_0007_6;2']} if sample_run else data
    
    raise Exception(f'unknown dataset: {dataset}')


def post_process(s, context):
    s = s.replace(u'\u00e9', 'e').replace(u'\u00ea', 'e').replace('\n\n', ' ')
    if context == 'visual':
        return s

    s = s.split('.')
    empty_sent_ids = []
    for idx in range(len(s)):
        s[idx] = s[idx].strip(' \" ').strip(' \' ').strip()
        s[idx] = s[idx].strip(' \' ').strip(' \" ').strip()
        if s[idx] == '':
            empty_sent_ids.append(idx)
    
    s = [s[idx] for idx in range(len(s)) if idx not in empty_sent_ids]
    return ' . '.join(s) + ' .'


def generate_story(dataset, seq, model, processor, context, prompt_id, num_sents):
    if context == 'visual':
        seq = Image.open(seq).convert('RGB')
        prompt = prompts_VC[prompt_id].replace('[#SENTS]', num_to_word[num_sents])
        inputs = processor(prompts_VC[prompt_id], seq, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=500, num_beams=1, do_sample=False)
        output = processor.decode(output[0], skip_special_tokens=True)
        
        return post_process(output.split('[/INST]')[1].strip(), context)

    if dataset != 'VIST':
        raise Exception('linguistic context settings not implemented for other datasets, refer to the paper')
    prompt_template = prompts_LC[prompt_id]
    story, prev_sent = '', ''
    for img_path in seq:
        image = Image.open(img_path + ('.jpg' if os.path.exists(img_path + '.jpg') else '.png')).convert('RGB')
        prompt_content = '' if context == 'no_context' else prev_sent if context == 'prev_sentence' else story
        prompt = prompt_template.format(prompt_content)

        inputs = processor(prompt, image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=500, num_beams=1, do_sample=False)
        output = processor.decode(output[0], skip_special_tokens=True)
        
        sentence = output.split('[/INST]')[1].strip()
        prev_sent, story = sentence, story + sentence + ' '

    return post_process(story.strip(), context)


def generate_stories(args):
    model, processor = load_model()
    data = load_data(args.dataset, args.data_path, sample_run=args.sample_run)
    
    stories = {}
    for sid, (imgs, _) in tqdm(data.items()):
        if args.dataset == 'VWP':
            imgs = imgs.split(',')
        seq = f'{args.data_path}/images/seqs/{sid}.png' if 'visual' in args.context else imgs
        try:
            stories[sid] = generate_story(args.dataset, seq, model, processor, args.context, args.prompt_id, num_sents=len(imgs))
        except Exception as e:
            print(f'could not generate story for {sid}: {e}')
            stories[sid] = 'N/A'

    return stories


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='LLaVA v1.6 for visual storytelling', 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ds', '--dataset', default='VIST', choices=['VIST', 'VWP'])
    parser.add_argument('-dp', '--data_path', default='data/vist',
                        help='path containing VIST images or composite image-sequences (concatenated images) for visual context')
    parser.add_argument('-sr', '--sample_run', action='store_true',
                        help='pass --sample_run to generate story for 1 sample')
    parser.add_argument('-st', '--save_to', default='data/stories',
                        help='path to save generated stories')
    parser.add_argument('-ct', '--context', default='visual', 
                        choices=['prev_sentence', 'all_sentences', 'visual', 'no_context'],
                        help='context to use when generating stories')
    parser.add_argument('-pt', '--prompt_id', default='P1', choices=['P1', 'P2', 'P3'],
                        help='prompt to use with the model')
    args = parser.parse_args()

    print(f'[ARGS] dataset: {args.dataset} test set, context: {args.context}, prompt_id: {args.prompt_id}, sample_run: {args.sample_run}\n')
    
    stories = generate_stories(args)
    
    print(stories)
    context_for_file_name = 'vc' if args.context == 'visual' else 'lc_as' if args.context.startswith('all') else 'lc_ps'
    save_to_file = f'{args.save_to}/{args.dataset.lower()}/llava_{context_for_file_name}_{args.prompt_id}.json'
    print(f'saving stories to {save_to_file}...', end='')
    with open(save_to_file, 'w') as fh:
        json.dump(stories, fh)
    fh.close()
    print('complete.\n')
