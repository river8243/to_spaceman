import logging
import re
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from bert_question_check import FORMAT, TextDataset

CHINESE_RE = r'[\u4e00-\u9fa5]'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

'''
TO-DO
[ ] 超過BERT長度上限(512)的句子如何處理？

'''

def _inference_single_batch(batch_data, model, device):
    input_ids, token_type_ids, attention_mask = [d.to(device) for d in batch_data[:3]]
    infos = batch_data[3]
    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )  # model output is single-length tuple which contains torch.tensor
    pred_labels_list = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
    pred_sentences, pred_punct_indices = _get_sentences(input_ids, infos, pred_labels_list, PUNCTUATIONS)
    return pred_sentences, pred_punct_indices


def load_model_and_tokenizer(device,
                             tokenizer_dir='/home/jovyan/if-beautiful-text/owen_dev/if_beautiful_text/cache_dir/bert-base-chinese',
                             qa_model_dir='/home/jovyan/wm-insur-call-qa/owen/bert_question_check/models/question_check/question_check_weight5_1_step2000_loss0.0966534224370944'):
    logging.info('Load bert tokenizer and QA model')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir, map_location=device)
    model = BertForSequenceClassification.from_pretrained(qa_model_dir, return_dict=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def inference(texts, tokenizer, model, device):
    batch_size = 4

    global MASK_INPUT_ID
    MASK_INPUT_ID = tokenizer.mask_token_id
    
    logging.info('Data transformation for model inference')
    dataset = TextDataset(tokenizer, texts, for_train=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.create_mini_batch)

    logging.info('Run prediction')
    pred_labels = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc='predict'):
            input_ids, token_type_ids, attention_mask = [d.to(device) for d in data[:3]]

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )

            pred_labels += outputs.logits.argmax(dim=-1).cpu().tolist()

    return pred_labels

def prepare_data(path):
    df = pd.read_csv(path, sep='\t')
    
    texts, trues = [], []
    for i in df.index:
        question = df.loc[i, 'question']
        answer = df.loc[i, 'answer']
        
        question = ''.join([q for q in question if re.match(CHINESE_RE, q)])
        answer = ''.join([a for a in answer if re.match(CHINESE_RE, a)])
        answer = '[UNK]' if not answer else answer

        texts.append([question, answer])
        label = 1 if df.loc[i, 'label'] == 5 else 0
        trues.append(label)
    return texts, trues

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'device: {device}')
    tokenizer, model = load_model_and_tokenizer(device)

    logging.info('Load text data')
    # Replace your data
    path = '/home/jovyan/wm-insur-call-qa/owen/data/qa_raw_copy.tsv'
    texts, trues = prepare_data(path)
#     texts = [
#         ['請問您本次投保繳交保費的資金來源是否為請問您是投保繳繳保費的資金來源是否為', '可'],
#         ['請問您投保石油城東分行的林怡辰從旁協助並由您本人親筆簽名嗎對好的那請問呃被保險人健康告知事項也是您本人親自填寫','是嗎'],
#     ['請問您投保石油城東分行的林怡辰從旁協助並由您本人親筆簽名嗎對好的那請問呃被保險人健康告知事項也是您本人親自填寫','嗯']]
    preds = inference(texts[7172:], tokenizer, model, device)
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, (text, true, pred) in enumerate(zip(texts[7172:], trues[7172:], preds)):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        print(tp, tn, fp, fn)
#         if true != pred:
#             print(i, text, true, pred)