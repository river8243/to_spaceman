# from fastapi import BackgroundTasks
from copy import copy
import os
from configobj import ConfigObj
import time
from utils import FileManager as fm
from utils import UtcTime
import json
from datetime import timezone
import pkg_resources
import socket
from SttModel import model
import multiprocessing as mp
from mlaas_tools.config_info import ConfigPass


class SttService:
    # 導入設定
    version = '0.1.0'
    log_set = ConfigPass()
    cfg = ConfigObj(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            'offline_stt_api_config.ini'))

    wav_storage_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        cfg['upload']['wav_storage_path'])

    wav_path_after_stt = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        cfg['stt']['wav_path_after_stt'])

    nj = int(cfg['stt']['num_jobs'])
    sleep_time = float(cfg['stt']['scheduler_sleep_time'])

    metadata_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        cfg['upload']['metadata_path'])

    stt_result_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        cfg['stt']['stt_res_storage_path'])

    model_path = model.absolute_model_path
    '''op_log_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        cfg['log']['log_storage_path'])'''


def create_stt_result_data(
        sentence,
        t_start,
        t_end,
        metadata_name,
        metadata,
        save_data=True):
    # 寫結果，辨識結果部份
    stt_result_data = copy(sentence)
    stt_result_data.update(metadata)
    stt_result_data['stt_service_version'] = SttService.version
    stt_result_data['recognizer_version'] = model.recognizer.version
    stt_result_data['vosk_version'] = pkg_resources.get_distribution(
        "vosk").version
    stt_result_data['recog_start_time'] = t_start
    stt_result_data['recog_end_time'] = t_end
    stt_result_data['recog_time'] = t_end - t_start
    stt_result_data['time_zone'] = str(timezone.utc)
    stt_result_data['am_version'] = SttService.model_path.rsplit('/', 1)[-1]
    stt_result_data['lm_version'] = SttService.model_path.rsplit('/', 1)[-1]
    stt_result_data['hostname'] = socket.gethostname()
    if save_data:
        with open(f'{SttService.stt_result_path}/{metadata_name}.json', 'w') as f:
            f.write(json.dumps(stt_result_data))
    SttService.log_set.logger.info(stt_result_data)
    return stt_result_data


def move_audio_and_cleanup_metadata(audio_file_path, audio_metadata_path):
    # 刪掉 metadata 檔案
    os.remove(audio_metadata_path)
    # 辨識完的 wav 要搬家
    os.rename(audio_file_path, os.path.join(
        SttService.wav_path_after_stt, audio_file_path.rsplit("/", 1)[-1]))


def single_audio_stt_inference(audio_file_path):
    metadata_name = audio_file_path.split('/', -1)[-1].rsplit('.', 1)[0]
    audio_metadata_path = f'{SttService.metadata_path}/{metadata_name}.json'
    if not os.path.isfile(audio_metadata_path):
        print(f'[請檢查] \nmetadata 檔案不存在，wav 路徑為 {audio_file_path}，忽略此檔辨識')
    t_start = UtcTime.get_current_utc_time()
    print(f'[開始辨識] \n於 {t_start} 辨識 {audio_file_path}')
    sentence = model.recognizer.recognize_wav_from_path(
        audio_file_path)
    t_end = UtcTime.get_current_utc_time()
    print(f'[辨識完成] \n{t_end} ：{sentence["text"]}，共花費 {t_end - t_start} 秒')
    with open(audio_metadata_path, 'r') as f:
        metadata = json.load(f)
    # 產生並寫結果
    stt_result_data = create_stt_result_data(
        sentence, t_start, t_end, metadata_name, metadata, save_data=True)
    #  搬辨識完的 wav 並 刪掉 metadata 檔案
    move_audio_and_cleanup_metadata(audio_file_path, audio_metadata_path)
    print(
        f'[落檔完成] \naudio: {audio_file_path} \nmetadata: {audio_metadata_path}')


def stt_inference(wav_list):
    '''
    Argument wav_list: a list of audio file paths.
    '''
    assert len(wav_list) > 0
    # 按造wav檔的順序排序
    wav_list.sort(key=os.path.getctime)
    for audio_file_path in wav_list:
        single_audio_stt_inference(audio_file_path)


def stt_inference_parallel(wav_list):
    '''
    Argument wav_list: a list of audio file paths.
    '''
    with mp.Pool(processes=SttService.nj) as pool:
        pool.map(single_audio_stt_inference, wav_list)


def stt_scheduler():
    # 開始主程式囉
    while True:
        # 掃檔、逐一辨識、辨識完搬檔
        wav_list = fm.list_all_file_type_in_dir(
            '\\.wav$', SttService.wav_storage_path)
        if len(wav_list) > 0:
            stt_inference_parallel(wav_list)
        else:
            time.sleep(SttService.sleep_time)


if __name__ == '__main__':
    stt_scheduler()
