import re
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from tqdm import tqdm
from python_speech_features import mfcc, logfbank
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import os

import sys
sys.path.insert(0,'/home/jovyan/wm-insur-call-qa/River/textgrid_open')
import textgrid_open
from fix_functions import time_fixer

def chunks(l, n):
    """
    Yield successive n-sized chunks from l
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

# people = ['esun', 'customer']
sl_s = 1
len_s = 1
para_dict = {1.5: {'max_total_context_test': 149,
                   'wav_long': 12000},
             1: {'max_total_context_test': 99,
                 'wav_long': 8000},
             0.5: {'max_total_context_test': 49,
                   'wav_long': 4000},
             0.2: {'max_total_context_test': 19,
                   'wav_long': 1600}}
max_total_context_test = para_dict[len_s]['max_total_context_test']

df_check = pd.read_csv('202012_speaker.tsv', sep='\t')

DIR_PATH = '/home/jovyan/wm-insur-call-qa/insur_data_202012'
dir_list = os.listdir(DIR_PATH)
dir_list = [name for name in dir_list if re.match(r'\d{8}', name)]

wm_dict = defaultdict(list)

for date in dir_list:
    for wav_name in tqdm(os.listdir(f'/home/jovyan/wm-insur-call-qa/insur_data_202012/{date}/')):
        wav_name_textgrid = wav_name.split('.')[0]
        try:
            textgrid = textgrid_open.get_textgrid_dict(f'/home/jovyan/wm-insur-call-qa/insur_data_202012_result/{date}/{wav_name_textgrid}.TextGrid')
        except:
            continue
        textgrid = {k.lower(): v for k, v in textgrid.items()}
        rate, wav = wavfile.read(f'/home/jovyan/wm-insur-call-qa/insur_data_202012/{date}/{wav_name}')

        tier_names_lower = [k for k, _ in textgrid.items()]    # 處理標註speaker name大小寫不一致

        if 'esun' not in tier_names_lower:
            continue
        else:
            tier_names_lower.remove('esun')

        esun_time = textgrid['esun']['times']
        for name in tier_names_lower:
            esun_time, _ = time_fixer(esun_time, textgrid[name]['times'])
        
        mfcc_list, class_list = [], []
            
        peo = 'esun'
        _min, _max = float('inf'), -float('inf')

        for time in esun_time:
            start_time = round(time[0], 2)
            end_time = round(time[1], 2)
            if (end_time - start_time) < len_s:
                pass
            else:
                for i in range(round((end_time - start_time) / sl_s) + 1):
                    try:
                        X_sample = mfcc(wav[round((start_time+i)*rate): round((start_time+i+len_s)*rate)], samplerate=rate, numcep=24, nfilt=26, nfft=1024)
                        _min = min(np.amin(X_sample),_min)
                        _max = max(np.amax(X_sample),_max)
                        for chunked_X_sample in list(chunks(X_sample, max_total_context_test)):
                            if len(chunked_X_sample) == max_total_context_test:
                                mfcc_list.append(chunked_X_sample)
                                class_list.append(peo)
                    except:
                        pass

        speaker = df_check.loc[df_check['pid'] == wav_name_textgrid, 'esun_name'].values[0]
        if mfcc_list:
            mfcc_list_f = (mfcc_list - _min) / (_max - _min)
            wm_dict[speaker].extend(mfcc_list_f)
with open('real_call_raw.pkl', 'wb') as f:
    pickle.dump(wm_dict, f)
for k, v in wm_dict.items():
    print(k, len(v))

    
