import logging

import pandas as pd

import speaker_diarization
from combine_stt_diarization import combine_stt_diarization
from time_locator import ctm_to_groupdata, get_time_span, get_report

MODEL_PATH = '/home/jovyan/wm-insur-call-qa/River/xvector_self/備份/total_model_1s'

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def pipeline(wav_file, ctm_path, questions):
    logging.info(f'Start process {wav_file}')
    sd = speaker_diarization.Speaker_Diarization(MODEL_PATH, wav_file, 10, 0.5, 5, 3)
    sd_result = sd.speaker_diarization_pipeline()    # 分段時間
    
    stt_df = ctm_to_groupdata(ctm_path, wav_file)

    logging.info(f'Combine sd and stt')
    combine_result = combine_stt_diarization(sd_result, stt_df[['start_time', 'stay_time', 'word']])

    result = get_report(speaker_df=combine_result, df=stt_df, questions=questions, max_cus_time=10, base_score=0.4)

if __name__ == "__main__":
    wav_file = '/home/jovyan/wm-insur-call-qa/testing_data/電訪-房貸壽險-法巴.wav'
    ctm_path = '/home/jovyan/wm-insur-call-qa/Lala/data/stt_result/eval.ctm'
    questions = []
    pipeline(wav_file, ctm_path, questions)