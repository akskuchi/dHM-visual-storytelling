import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import json
import os
import argparse
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompts_VC = {
    'P1': '[INST] <image>\nWrite a story using exactly five sentences for this image sequence. Do not use more than five sentences. [/INST]',
    'P2': '[INST] <image>\nGenerate a story consisting of five sentences for this image sequence. Use only five sentences and not more. [/INST]',
    'P3': '[INST] <image>\nOutput a story about this sequence of images using only five sentences. Make sure the story does not include more than five sentences. [/INST]'
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


def load_vist_metadata(vist_data_path, sample_run=True):
    data = {}
    imgs, txts = [], []
    annos = json.load(open(f'{vist_data_path}/sis/test.story-in-sequence.json'))['annotations']
    for idx in range(len(annos)):
        imgs.append(f'{vist_data_path}/filtered_test/{annos[idx][0]["photo_flickr_id"]}')
        txts.append(annos[idx][0]['text'])
        if len(imgs) == 5:
            data[annos[idx][0]['story_id']] = (imgs, txts)
            imgs, txts = [], []

    return {'49620': data['49620']} if sample_run else data


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


def generate_story(seq, model, processor, context, prompt_id):
    if context == 'visual':
        seq = Image.open(seq).convert('RGB')
        inputs = processor(prompts_VC[prompt_id], seq, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=500, num_beams=1, do_sample=False)
        output = processor.decode(output[0], skip_special_tokens=True)
        
        return post_process(output.split('[/INST]')[1].strip(), context)

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
    data = load_vist_metadata(args.vist_data_path, sample_run=args.sample_run)
    
    stories = {}
    for sid, (seq, _) in tqdm(data.items()):
        seq = f'{args.vist_data_path}/test/strips/{sid}.png' if 'visual' in args.context else seq
        try:
            stories[sid] = generate_story(seq, model, processor, args.context, args.prompt_id)
        except Exception as e:
            print(f'could not generate story for {sid}: {e}')
            stories[sid] = 'N/A'

    return stories


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='zero-shot LLaVA v1.6 stories for VIST test set', 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--vist_data_path', default='data/vist-data',
                        help='path containing VIST images or image-sequence strips (for visual context)')
    parser.add_argument('--sample_run', action='store_true',
                        help='pass --sample_run to test on 1 sample')
    parser.add_argument('--save_to', default='data/stories',
                        help='path to save generated stories')
    parser.add_argument('--context', default='no_context', 
                        choices=['no_context', 'prev_sentence', 'all_sentences', 'visual'],
                        help='context to use when generating stories')
    parser.add_argument('--prompt_id', default='P1', choices=['P1', 'P2', 'P3'],
                        help='prompt ID')
    args = parser.parse_args()

    print(f'LLaVA for VIST test set context: {args.context}, prompt_id: {args.prompt_id}, sample_run: {args.sample_run}\n')
    
    stories = generate_stories(args)
    
    save_to_file = f'{args.save_to}/llava_vist_{args.context}_{args.prompt_id}.json'
    print(f'saving stories to {save_to_file}...', end='')
    with open(save_to_file, 'w') as fh:
        json.dump(stories, fh)
    fh.close()
    print('complete.\n')
