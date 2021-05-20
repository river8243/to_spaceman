import os
import re
import logging

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import tokenize_and_map, RunningAverage

CHINESE_RE = r'[\u4e00-\u9fa5]'
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, trues=None,
                 pad_token_label_id=0, max_length=512, for_train=True):
        self.tokenizer = tokenizer
        self.texts = texts
        self.trues = trues
        self.pad_token_label_id = pad_token_label_id
        self.max_length = max_length
        self.for_train = for_train

    def __getitem__(self, idx):
        q_texts = self.texts[idx]

        processed_tokens = ['[CLS]'] 
        for text in q_texts:
            tokens, text2token, token2text = tokenize_and_map(self.tokenizer, text)

            cut_index = self.max_length - 50
            if cut_index < len(tokens):
                cut_text_index = text2token.index(cut_index)
                tokens = tokens[:cut_index]
                            
            processed_tokens += tokens + ['[SEP]']
            
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))
        token_type_ids = torch.tensor([0] * len(processed_tokens))
        attention_mask = torch.tensor([1] * len(processed_tokens))

        outputs = (input_ids, token_type_ids, attention_mask, )

        if self.for_train:
            true = self.trues[idx]
            label = torch.tensor(true)
            outputs += (label, )

        info = {
            'tokens': tokens,
            'text': q_texts, # [ ] 並非實際進BERT的句子(512上限)
        }
        outputs += (info, )
        return outputs

    def __len__(self):
        return len(self.texts)

    def create_mini_batch(self, samples):
        outputs = list(zip(*samples))

        # zero pad 到同一序列長度
        input_ids = pad_sequence(outputs[0], batch_first=True)
        token_type_ids = pad_sequence(outputs[1], batch_first=True)
        attention_mask = pad_sequence(outputs[2], batch_first=True)

        batch_output = (input_ids, token_type_ids, attention_mask)
    
        if self.for_train:
            labels = torch.stack(outputs[3])
            batch_output += (labels, )
        else:
            infos = outputs[3]
            batch_output += (infos, )

        return batch_output

def train_batch(model, data, optimizer, device):
    model.train()
    input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data]

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, valid_loader):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    loss_averager = RunningAverage()
    acc_averager = RunningAverage()

    tp, fp, fn = 0, 0, 0    
    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluate'):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data]

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss_averager.add(outputs.loss.item())
            
            corrects = (outputs.logits.argmax(dim=-1) == labels).cpu().tolist()
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

            for label, pred in zip(labels, preds):
                tp += 1 if (label, pred) == (0, 0) else 0
                fp += 1 if (label, pred) == (1, 0) else 0
                fn += 1 if (label, pred) == (0, 1) else 0
            
            acc_averager.add_all(corrects)
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    f1 = 2 / (1 / precision + 1 / recall) if precision and recall else None
    
    evaluation = {
        'loss': loss_averager.get(), 
        'accuracy':acc_averager.get(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return evaluation

def bert_main(texts, trues,
              model_dir='/home/jovyan/if-beautiful-text/owen_dev/if_beautiful_text/cache_dir/bert-base-chinese',
              save_dir='./models/question_check/'):
    lr = 0.00001
    train_batch_size = 8
    evaluate_batch_size = 64
    max_iter = 100000
    show_train_per_iter = 100
    show_eval_per_iter = 100
    save_per_iter = 1000
    cpu_workers = 4
    checkpoint_folder = None

    assert save_per_iter % show_eval_per_iter == 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'device: {device}')

    tokenizer = BertTokenizer.from_pretrained(model_dir)

    global SKIP_TOKEN_IDS, SKIP_TOKENS
    SKIP_TOKEN_IDS = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    SKIP_TOKENS = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]

    if not checkpoint_folder:
        model = BertForSequenceClassification.from_pretrained(
            model_dir, 
            return_dict=True,
            num_labels=2)
    else:
        model = BertForSequenceClassification.from_pretrained(checkpoint_folder)

    model.to(device)

    dataset = TextDataset(tokenizer, texts, trues)

    CUT_RATIO = 0.8
    train_size = int(CUT_RATIO * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # t_texts, t_trues = prepare_data('/home/jovyan/wm-insur-call-qa/owen/training_data_generated_train_0119.tsv')
    # train_dataset = TextDataset(tokenizer, t_texts, t_trues)

    # v_texts, v_trues = prepare_data('/home/jovyan/wm-insur-call-qa/owen/training_data_generated_valid_0119.tsv')
    # valid_dataset = TextDataset(tokenizer, v_texts, v_trues)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        collate_fn=dataset.create_mini_batch,
        shuffle=True,
        num_workers=cpu_workers)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=evaluate_batch_size,
        collate_fn=dataset.create_mini_batch,
        shuffle=True,
        num_workers=cpu_workers)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    i = 1
    is_running = True
    loss_averager = RunningAverage()
    while is_running:
        for train_data in train_loader:
            loss = train_batch(model, train_data, optimizer, device)
            loss_averager.add(loss)

            if i % show_train_per_iter == 0:
                logging.info('train_loss [{iter}]: {train_loss}'.format(
                    iter=i, train_loss=loss_averager.get()))
                loss_averager.flush()

            if i % show_eval_per_iter == 0:
                evaluation = evaluate(model, valid_loader)
                logging.info('valid_evaluation: loss={loss}, accuracy={accuracy}, '
                             'precision={precision}, recall={recall}, f1={f1}'
                             .format(**evaluation))

            # if i % save_per_iter == 0:
            #     eval_loss = evaluation['loss']
            #     path = os.path.join(save_dir, f'question_check_step{i}_loss{eval_loss}/')
            #     logging.info(f'Save model at {path}')
            #     model.save_pretrained(path)

            if i == max_iter:
                is_running = False
                break
            i += 1

        scheduler.step()

def prepare_data(path):
    df = pd.read_csv(path, sep='\t')
    
    texts, trues = [], []
    for i in df.index:
        question = df.loc[i, 'question']
        answer = df.loc[i, 'answer']
        
        question = ''.join([q for q in question if re.match(CHINESE_RE, q)])
        answer = ''.join([a for a in answer if re.match(CHINESE_RE, a)])

        texts.append([question, answer])
        label = 1 if df.loc[i, 'label'] == 5 else 0
        trues.append(label)
    return texts, trues


if __name__ == '__main__':
    path = '/home/jovyan/wm-insur-call-qa/owen/data/qa_raw.tsv'
    texts, trues = prepare_data(path)
    
    # Test Data
    # texts = [['您好！這裡是玉山銀行總行個金處/OO分行/OO消金中心，敝姓O，員工編號OOOOO，請問是○○○先生/小姐本人嗎？', '嘿我是'],
    # ['請問您的出生年月日是?', '六十一月十七']]
    # trues = [3, 2]

    bert_main(texts, trues)
