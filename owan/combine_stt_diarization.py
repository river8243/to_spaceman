import pandas as pd


def overlap_range(row, start_time, end_time):
    overlap_range = min(row.end, end_time) - max(row.start, start_time)
    overlap_ratio = overlap_range / (end_time - start_time)
    return overlap_ratio

def combine_stt_diarization(diariz_list, ctm_df):
    """
    ctm_df: pd.DataFrame
            must has columns ['start_time', 'stay_time', 'word']
    """
    # 找出音檔結尾時間
    stt_end_time = ctm_df.start_time.iloc[-1] + ctm_df.stay_time.iloc[-1]

    # 併上音檔結束時間(如果最後一個詞的結束時間比最後的轉換時間還大的話)
    if stt_end_time > diariz_list[-1]:
        diariz_list.append(stt_end_time)

    # 做出語者分離時間表
    diariz_dict = {'start': diariz_list[: -1],
                   'end': diariz_list[1:],
                   'speaker': [0 if i%2==0 else 1 for i in range(len(diariz_list)-1)]}
    diariz_df = pd.DataFrame(diariz_dict, columns = ['start', 'end', 'speaker'])
    diariz_df['word_idx'] = [[] for i in range(len(diariz_df))]

    # combine ctm_rslt to diarize_rslt 
    for i in range(len(ctm_df)):
        start_time = ctm_df['start_time'][i]
        end_time = start_time + ctm_df['stay_time'][i]

        match_list = diariz_df[(diariz_df['start'] <= start_time) & (diariz_df['end'] >= end_time)].index.to_list()

        if match_list!=[]:
            append_idx = match_list[0]
            diariz_df['word_idx'][append_idx].append(ctm_df['word'][i])
        else:
            overlap_list = diariz_df.apply(lambda x: overlap_range(x, start_time, end_time), axis=1)
            match_list = overlap_list[overlap_list > 0].index.to_list()
            if match_list != []:
                # append_idx = max(match_list)
                # diariz_df['word_idx'][append_idx].append(ctm_df['word'][i])
                for match_idx in match_list:
                    diariz_df['word_idx'][match_idx].append(ctm_df['word'][i])

    return diariz_df

sd_result = [0, 11.2, 11.6, 12.95, 13.3, 14.1, 15.15, 19.3, 19.75, 20.65, 21.5, 31.5, 32.3, 32.75, 33.35, 35.1, 35.2, 37.7, 38.0, 39.5, 40.0, 40.7, 41.95, 44.05, 44.25, 45.45, 46.3, 48.6, 49.85, 51.15, 51.6, 60.25, 62.3, 64.2, 64.7, 73.7, 74.7, 77.0, 77.5, 82.35, 83.1, 85.0, 85.9, 90.6, 94.25, 96.0, 99.15, 105.3, 107.4, 114.3, 114.6, 121.4, 121.55, 132.2, 132.8, 141.3, 141.9, 148.9, 149.4, 150.3, 150.7, 151.9, 152.4, 157.8, 158.35, 160.35, 160.75, 161.8, 162.85, 168.2, 168.75, 175.1, 176.35, 177.55, 182.75, 183.9, 184.95, 186.25, 188.0, 193.5, 195.45, 196.55, 197.05, 197.95, 198.4, 199.95, 201.4, 203.35, 206.1]
from time_locator import ctm_to_groupdata
data_path = '/home/jovyan/wm-insur-call-qa/Lala/data/stt_result/eval.ctm'
group_name = '/home/jovyan/wm-insur-call-qa/testing_data/電訪-房貸壽險-法巴.wav'
stt_df = ctm_to_groupdata(data_path, group_name)
combine_result = combine_stt_diarization(sd_result, stt_df[['start_time', 'stay_time', 'word']])