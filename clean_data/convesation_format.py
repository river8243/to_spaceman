import re
import os
import json

from tqdm import tqdm

from praat_textgrid_methods import PraatTextgrid

DIR_PATH = '/home/jovyan/wm-insur-call-qa/insur_data_202012_result'
SAVE_PATH = '/home/jovyan/wm-insur-call-qa/insur_data_202012_conversation'


dir_list = os.listdir(DIR_PATH)
dir_list = [name for name in dir_list if re.match(r'\d{8}', name)]

for date in dir_list:
    print(f'Start process {date}')
    conversations = {}
    file_list = os.listdir(f'{DIR_PATH}/{date}')
    
    for file_name in tqdm(file_list):
        print(file_name)
        # print(file_name)
        if not 'TextGrid' in file_name:
            continue
        test_read_tg = os.path.join(f'{DIR_PATH}/{date}/{file_name}')
        textgrid, tier_names = PraatTextgrid.get_tiernames_from_tgfile(test_read_tg)
        tier_names_lower = [name.lower() for name in tier_names]
        if not all(i in tier_names_lower for i in ['esun', 'customer']):
            continue

        esun_name = tier_names[tier_names_lower.index('esun')]
        cust_name = tier_names[tier_names_lower.index('customer')]        
        esun_annotation = PraatTextgrid.read_tier_from_tg(textgrid, esun_name).annotations
        cust_annotation = PraatTextgrid.read_tier_from_tg(textgrid, cust_name).annotations

        all_texts = [('e', annotation.start_time, annotation.text) for annotation in esun_annotation] \
                    + [('c', annotation.start_time, annotation.text) for annotation in cust_annotation]
        all_texts_sorted = sorted(all_texts, key=lambda tup: tup[1])

        esun_sent, cust_sent, last_speaker = '', '', ''
        conversation = []
        while all_texts_sorted:
            text = all_texts_sorted.pop(0)
            speaker = text[0]        
            if last_speaker == 'c' and speaker == 'e' and esun_sent:
                conversation.append((esun_sent.strip(','), cust_sent.strip(',')))
                esun_sent, cust_sent = '', ''

            if speaker == 'e':
                esun_sent += text[2] + ','
            elif speaker == 'c':
                cust_sent += text[2] + ','

            last_speaker = speaker
        if esun_sent and cust_sent:
            conversation.append((esun_sent.strip(','), cust_sent.strip(',')))
        conversations[file_name.split('.')[0]] = conversation
    with open(f'{SAVE_PATH}/{date}_conversations.json', 'w') as fw:
        fw.write(json.dumps(conversations))