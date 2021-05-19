import copy
import configparser
import json
from utils import UtcTime
from uuid import UUID
import os
from configobj import ConfigObj
import subprocess
from mlaas_tools import APIBase


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        return json.JSONEncoder.default(self, obj)


class Operation(APIBase):  # 統一名稱為 Operation
    def __init__(self):
        super().__init__()  # 繼承 APIBase 這個 class
        # 導入設定
        cfg = ConfigObj(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'offline_stt_api_config.ini'))
        self.wav_storage_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            cfg['upload']['wav_storage_path'])
        self.metadata_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            cfg['upload']['metadata_path'])

        self.wav_path_after_stt = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            cfg['stt']['wav_path_after_stt'])
        self.stt_result_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            cfg['stt']['stt_res_storage_path'])

        # 確使暫存資料夾存在:
        for directory in [
                self.wav_storage_path,
                self.wav_path_after_stt,
                self.metadata_path,
                self.stt_result_path]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        self.proc = subprocess.Popen("python stt_service.py", shell=True)

        self.valid_audio_format = set(cfg['stt']['valid_audio_format'])

    def have_warning_inputs(self, input_data):
        have_warning = False
        input_arg_list = [
            'request_id',
            'business_unit',
            'agent_id',
            'agent_number',
            'call_duration',
            'client_number',
            'dial_in_time',
            'hang_up_time'
        ]
        warning_args = []
        for key in input_data.keys():
            if key not in input_arg_list:
                have_warning = True
                warning_args.append(key)
        for arg in input_arg_list:
            if arg not in input_data.keys():
                have_warning = True
                warning_args.append(arg)
        return have_warning, warning_args

    def _land_a_file(self, sample_rate=16000):
        '''
        input:
          reques_file : werkzeug.datastructures.FileStorage
          landing_name : string
        output:
          bool // True if success else False
        '''
        ext = os.path.splitext(self.file.filename)[1][1:]
        if ext not in self.valid_audio_format:
            return False

        landing_name = f'{self.wav_storage_path}/{self.trace_id}.wav'
        tmp_name = f'{self.wav_storage_path}/{self.trace_id}_tmp.{ext}'
        self.file.save(tmp_name)

        process = subprocess.Popen(
            f'sox -t {ext} {tmp_name} -r {sample_rate} -c 1 -b 16 -t wav {landing_name}; rm {tmp_name}',
            shell=True)

        return not bool(process.returncode)

    def run(self, input_data):
        upload_time = UtcTime.get_current_utc_time()
        input_data['request_id'] = self.request_id
        input_data['business_unit'] = self.business_unit
        # 取得 input_data 內部資料
        # construct response:
        have_warning, warning_inputs = self.have_warning_inputs(input_data)
        stt_service_is_alive = (self.proc.poll() is None)
        self.response = {}
        if stt_service_is_alive:
            if not have_warning:
                self.response['response_code'] = 201
                self.response['response_message'] = {'message': 'success'}
            else:
                self.response['response_code'] = 202
                self.response['response_message'] = {
                    'warning_inputs': warning_inputs}
        else:
            self.response['response_code'] = 499
            self.response['response_message'] = {
                "error_message": "Unexpected error occurred."}

        # land a file
        try:
            self._land_a_file()
        except BaseException:
            self.response['response_code'] = 401
        self.response['upload_filename'] = self.file.filename
        # generate elastic search log

        es_log = copy.copy(input_data)
        es_log['response_code'] = self.response['response_code']
        es_log['response_message'] = self.response['response_message']
        es_log['upload_filename'] = self.response['upload_filename']
        es_log['trace_id'] = self.trace_id
        es_log['upload_api_version'] = '100'  # 代表上傳 api 的版本
        es_log['upload_time'] = upload_time  # 上傳此檔案時間，UTC timestamp

        with open(f'{self.metadata_path}/{self.trace_id}.json', 'w') as fp:
            json.dump(es_log, fp, cls=UUIDEncoder)
        return self.response


'''
mlaas會自己回傳的部分:
- [v] request_id: str  # 請求識別碼，就跟丟過來的一樣
- [v] trace_id: str  # 追蹤識別碼
- [v] request_time: datetime.datetime  # 請求時間，UTC timestamp
- [v] response_time: datetime.datetime  # 回應時間，UTC timestamp
- [v] duration_time: datetime.datetime  # 回應時間 - 請求時間


- response_code: str  # 回覆狀態碼
    所有可能的選項:
    [v] "201": 正常
    [v] "202": Warning 訊息: 欄位異常 (e.g., 要求A, B欄位，給了A, B, C欄位，多了欄位C)
    "401": 欄位值不符合規範
    "405": 辨識系統忙碌中，請稍候重打 (data folder有超過1000個wav)
    "499": 未預期錯誤
- response_message: dict  # 回覆狀態碼內容
    所有可能的選項:
    [v] "201": {"message": "success"}
    [v] "202": {"warning_inputs": ["cust_no", "date"]}
    "401": {"error_inputs": ["business_unit", " dial_in_time "]}
    "405": {"error_message": "system busy, try it later."}
    "499": {"error_message": "Unexpected error occurred."}
'''
