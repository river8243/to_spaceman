from difflib import SequenceMatcher
import ast
import re
import pandas as pd
import math

def ctm_to_groupdata(data_path, group_name):
    # load stt results
    ctm_df = pd.read_csv(data_path, sep=' ', header=None)
    ctm_df = ctm_df.iloc[:, :-1]
    ctm_df.columns = ['file_name', '1', 'start_time', 'stay_time', 'word']
    selectors = ctm_df.groupby('file_name')
    group_data = selectors.get_group(group_name)
    group_data = group_data.reset_index(drop=True)
    return group_data

# similarity function
def similar(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

# 拿到最高機率time span後再進行細部比對
def get_similar_tokens(tokens, sample_str, max_score):
    _str = ''.join(i.word for i in tokens)

    start_idx = 0
    temp = tokens
    for i in range(len(tokens)):
        start_idx += len(tokens[i].word)
        score = similar(_str[start_idx:], sample_str)
        if  score >= max_score :
            max_score = score
            temp = tokens[i+1:]
        if score < 0.3:
            break
    tokens = temp        
    _str = ''.join(i.word for i in tokens)
    end_idx = len(_str)

    for i in range(len(tokens)-1, 0, -1):
        end_idx -= len(tokens[i].word)
        score = similar(_str[0:end_idx], sample_str)
        if  score >= max_score:
            max_score = score
            tokens = tokens[: i]
        elif score < 0.3:
            break
    return tokens, max_score


# 輸入Dataframe和要搜尋的字 
#  return dict
#  start_time:  float : 起始秒數
#  end_time:  float : 起始秒數
#  tokens: list of pd.Series  最接近搜索字串的token list
#  max_score: 最高相似分數
#  token_text : list of string : 問題對應的tts辨識結果
def get_time_span(df, ori_question):
    sign = ['，', '。', '?', '？', '、', '!', '！', '/', 'O', '○', ' ', ',', '.', '_', '(', ')', '（', '）', '\n']

    sample_str = ori_question
    for s in sign:
        sample_str = sample_str.replace(s, '')
    items = [row for index, row in df.iterrows()]
    similar_tokens = []
    max_score = 0.3
    for i in range(len(items) - len(sample_str)):
        _str = ''.join(i[-1] for i in items[i: i+len(sample_str)])
        score = similar(_str, sample_str)
        if score >= max_score:
            max_score = score
            similar_tokens.append([items[i: i+len(sample_str)], score])

    if len(similar_tokens) == 0:
        return {'start_time': 0, 'end_time': 0, 'tokens':[], 'max_score': 0, 'tokens_text': [], 'ori_question': ori_question}            

    try:
        tokens = similar_tokens[-1][0]
        max_score = similar_tokens[-1][1]
        tokens, max_score = get_similar_tokens(tokens, sample_str, max_score)
        return {'start_time': tokens[0].start_time, 'end_time': tokens[-1].start_time+tokens[-1].stay_time,\
                'tokens': tokens, 'max_score': max_score, 'tokens_text': [i.word for i in tokens], 'ori_question': ori_question}
    except:
        return {'start_time': 0, 'end_time': 0, 'tokens': [], 'max_score': max_score, 'tokens_text': [], 'ori_question': ori_question}          


# 判斷回答是否正面
# 可自己決定template
def is_positive(tokens, template=None):  
    if template == None:
        template = '好的|了解|知道|是|好|對'
    
    for token in tokens:
        combine_token = ''.join(token)
        if len(re.findall(template, combine_token)):
            return 1
    return 0



# 拿到報告
# speaker_df [start, end, speaker, word_idx] 類似對話腳本的dataframe
# df [file_name, start_time, stay_time, word] STT的ctm檔
# question: 問題list
# max_cus_time 當Q1結束 到 Q2開始時間超過這個長度 就會自動減短到 max_cus_times的間距 避免問題對調or遺失問題
# max_score 辨識和問題的相似度超過max_score才算通過檢驗
def get_report(speaker_df, df, questions, max_cus_time=20, base_score=0.4):
    # decide who is cleark
    cleark_id, customer_id = 0, 0
    speaker_df = speaker_df.copy()
    speaker_df['speaktime'] = speaker_df.end - speaker_df.start 
    stime = speaker_df.groupby('speaker').speaktime.sum()

    temp_record = []
    if stime[1] > stime[0]:
        cleark_id = 1
    else:
        customer_id = 1

    # find the question  time span (if score > base_score)
    q_result = [get_time_span(df, q) for q in questions]
    q_result = sorted(q_result, key=lambda x: x['start_time'])

    # record  time info
    for i, qs in enumerate(q_result):
        # add question and time start & end
        temp_record.append([qs['start_time'], qs['end_time'], qs['max_score'], qs['ori_question'], ''.join(qs['tokens_text'])])

    for i in range(len(q_result)):
        if i+1 == len(q_result):
            end_time = q_result[i]['end_time'] + max_cus_time
        else:
            end_time = q_result[i+1]['start_time'] + 0.5

        start_time = int(q_result[i]['end_time']-0.5)

        #if end_time-start_time > max limit it to +max_cus_time
        if end_time - start_time > max_cus_time:
            end_time = q_result[i]['end_time'] + max_cus_time

        cus_range = speaker_df.loc[(speaker_df['start'] >= start_time) &
                                   (speaker_df['end'] <= end_time) &
                                   (speaker_df['speaker'] == customer_id)]

        # 每個token的文字
        items = [''.join(j[-1].word_idx) for j in cus_range.iterrows()]
        # 每個token的時間標記
        token_time = [(j[-1].start, j[-1].end) for j in cus_range.iterrows()]
        pass_result = is_positive(items)
        if q_result[i]['max_score'] < base_score:
            pass_result = 0

        if len(token_time):
            if temp_record[i][0]==0 and temp_record[i][1]==0:
                temp_record[i] += [0, 0, [], 0]
            else:
                temp_record[i] += [token_time[0][0], token_time[-1][1], items, pass_result]
        else:
            temp_record[i] += [0, 0, items, 0]

    report = pd.DataFrame(temp_record, columns=["q_start_time", "q_end_time", "score", "question", "recognize_result",
                                                "reply_start_time", "reply_end_time", "reply", "pass"])

    return report
