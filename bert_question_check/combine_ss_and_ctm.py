from difflib import SequenceMatcher

import pandas as pd
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
import librosa

def ctm_to_groupdata(data_path, group_name):
    # load stt results
    ctm_df = pd.read_csv(data_path, sep=' ', header=None)
    ctm_df = ctm_df.iloc[:, :-1]
    ctm_df.columns = ['file_name', '1', 'start_time', 'stay_time', 'word']
    selectors = ctm_df.groupby('file_name')
    group_data = selectors.get_group(group_name)
    group_data = group_data.reset_index(drop=True)
    return group_data

def get_times(group_data):
    times = []
    for i in group_data.index:
        start = group_data.loc[i, 'start_time']
        end = (start + group_data.loc[i, 'stay_time']).round(2)
        if times:
            if start == times[-1][1]:
                last_start, last_end = times.pop()
                start = last_start
        times.append((start, end))
    return times

def _prepare_calculate_rms(wav_path, rate):
    y, _ = librosa.load(wav_path, sr=rate)
    rms_sample = librosa.feature.rms(y=y)
    _max, _min = np.amax(rms_sample), np.amin(rms_sample)
    return y, _max, _min 

def get_sentences_rms(origin_ctm, wav_path_a, wav_path_b, rate=8000):
    # 產出行員與顧客之個人逐字稿
    y_a, _max_a, _min_a = _prepare_calculate_rms(wav_path_a, rate)
    y_b, _max_b, _min_b = _prepare_calculate_rms(wav_path_b, rate)

    a_sentences, a_starts, a_ends, b_sentences, b_starts, b_ends = [], [], [], [], [], []
    a_seg, a_time, b_seg, b_time = [], [], [], []

    a_times, b_times = [], []
    for i in origin_ctm.index:
        start = origin_ctm.loc[i, 'start_time']
        end = (start + origin_ctm.loc[i, 'stay_time']).round(2)
        text = origin_ctm.loc[i, 'word']
        if text == '<SIL>':
            continue

        rms_a = librosa.feature.rms(y=y_a[int(start*rate): int(end*rate)])
        rms_a = (rms_a - _min_a) / (_max_a - _min_a)
        rms_b = librosa.feature.rms(y=y_b[int(start*rate): int(end*rate)])
        rms_b = (rms_b - _min_b) / (_max_b - _min_b)
        
        if np.mean(rms_a) >= np.mean(rms_b):
            a_seg += [text]
            a_time += [start, end]
        elif a_seg:
            a_sentences.append(' '.join(a_seg))
            a_starts.append(a_time[0])
            a_ends.append(a_time[-1]) 
            a_seg, a_time = [], []
            
        if np.mean(rms_a) <= np.mean(rms_b):
            b_seg += [text]
            b_time += [start, end]
        elif b_seg:
            b_sentences.append(' '.join(b_seg))
            b_starts.append(b_time[0])
            b_ends.append(b_time[-1]) 
            b_seg, b_time = [], []
  
        
    a_df = pd.DataFrame({'start': a_starts, 'end': a_ends, 'sentence': a_sentences})
    b_df = pd.DataFrame({'start': b_starts, 'end': b_ends, 'sentence': b_sentences})
    return a_df, b_df 
group_1_asr, group_2_asr = get_sentences_rms(origin_ctm, group_1_name, group_2_name)

def get_report(customer_ctm, origin_ctm, questions, base_score=0.4):
    # find the question  time span (if score > base_score)
    q_result = [get_time_span(origin_ctm, q) for q in questions]
    q_result = sorted(q_result, key=lambda x: x['start_time'])

    # record  time info
    temp_record = []
    for i, qs in enumerate(q_result):
        # add question and time start & end
        temp_record.append([qs['start_time'], qs['end_time'], qs['max_score'], qs['ori_question'], ''.join(qs['tokens_text'])])

    for i in range(len(temp_record)):
        q_end = q_result[i]['end_time']
#         next_q_start = q_result[i]['end_time']
        sentences = group_2_asr.loc[customer_ctm['start'].apply(lambda x: q_end-2 < x < q_end+5), :]
        ans = sentences.values[0].tolist() if not sentences.empty else None
        if ans and (temp_record[i][2] != 0):
            ans.append(''.join(ans.pop().split()))
            temp_record[i] += ans
        else:
            temp_record[i] += [0, 0, None]
    report = pd.DataFrame(temp_record, columns=["q_start_time", "q_end_time", "score", "question", "recognize_result",
                                                "reply_start_time", "reply_end_time", "reply"])
    return report        

if __name__=='__main__':
    wav = '電訪-傳統型台幣-中壽'
    origin_name = f'/home/jovyan/wm-insur-call-qa/eric/speaker-separation/test_zone/test_result/{wav}.wav'
    origin_ctm = ctm_to_groupdata('/home/jovyan/exchanging-pool/to_owen/func_asr/stt_result/ctm/ctm', origin_name)

    group_1_name = f'/home/jovyan/wm-insur-call-qa/eric/speaker-separation/test_zone/test_result/DPRNN1-{wav}.wav'
    group_1_ctm = ctm_to_groupdata('/home/jovyan/exchanging-pool/to_owen/func_asr/stt_result/ctm/ctm', group_1_name)
    group_1_times = get_times(group_1_ctm)

    group_2_name = f'/home/jovyan/wm-insur-call-qa/eric/speaker-separation/test_zone/test_result/DPRNN2-{wav}.wav'
    group_2_ctm = ctm_to_groupdata('/home/jovyan/exchanging-pool/to_owen/func_asr/stt_result/ctm/ctm', group_2_name)
    group_2_times = get_times(group_2_ctm)
    
    group_1_asr, group_2_asr = get_sentences_rms(origin_ctm, group_1_name, group_2_name)
    
    # 辨識顧客語音
    times_1 = sum([e - s for s, e in zip(group_1_asr.start, group_1_asr.end)])
    times_2 = sum([e - s for s, e in zip(group_2_asr.start, group_2_asr.end)])
    customer_ctm = group_1_asr if times_1 < times_2 else group_2_asr   # customer is group 1 or 2
    customer_ctm

    questions = ["""您好！這裡是玉山銀行總行個金處/OO分行/OO消金中心，
               敝姓O，員工編號OOOOO，請問是○○○先生/小姐本人嗎？""",
            '感謝您近期透過本行投保○○人壽○○○，繳費年期為O年，依照保險法令的要求，為保障您的權益，稍後電話訪問內容將會全程錄音，請問您同意嗎？'
            '為維護您的資料安全，這裡簡單跟您核對基本資料，您的身分證字號是，請問後三碼是？',
            '請問您的出生年月日是?',
            '請問您是否知道本次購買的是○○人壽的保險，不是存款，如果辦理解約將可能只領回部分已繳保費？',
            '請問您投保時，是否皆由○○消金中心的○○○，在旁邊協助，並由您本人○○○親自簽名，且被保險人之健康告知事項皆由您確認後親自填寫？',
            '請問○○消金中心的○○○是否有向您說明產品內容，並確認符合您的需求？',
            '請問招攬人員是否有提供您一次繳清與分期繳等不同繳費方式選擇？',
            '請問您本次投保繳交保費的資金來源是否為',
            """請問您是否已事先審慎評估自身財務狀況與風險承受能力，
               並願承擔因財務槓桿操作方式所面臨的風險及辦理保單解約轉投保之權益損失，
               除辦理貸款或保單借款需支付本金及利息外，
               還有該產品可能發生之相關風險及最大可能損失，
               且本行人員並未鼓勵或勸誘以辦理貸款、保單借款、保單解約/保單終止及定存解約之方式購買保險，
               請問您是否已瞭解？""",
            '與您確認，本保單之規劃您是否已確實瞭解投保目的、保險需求，並經綜合考量財務狀況以及付費能力，且不影響您的日常支出？',
            '與您再次確認上述投保內容和本次貸款並沒有搭售或不當行銷的情形發生，請問是否正確?',
            '請問您本次辦理貸款及保險，是否有新申請玉山網路銀行？']    
    
    get_report(customer_ctm, origin_ctm, questions)
