"""
api test
"""
import sys
import requests
from mlaas_tools.api_test_config import APITestInfo
from datetime import datetime
from datetime import timezone
import json


def construct_api_payload(env_now='Aicloud', business_unit='IF_C170'):

    dial_in_datetime = datetime(2021, 4, 19, 8, 40)
    hang_up_datetime = datetime(2021, 4, 19, 8, 50)
    call_duration = (hang_up_datetime - dial_in_datetime).total_seconds()

    inputs = {
        'client_number': '0912345678',  # 顧客電話號碼，顧客撥入電話碼，若匿名請留空
        'agent_id': 'esb00000',  # 客服員編，用以識別接聽人員
        'agent_number': '0000',  # 客服分機號碼，用以識別接聽座席
        # 電話撥入時間，UTC timestamp
        'dial_in_time': dial_in_datetime.replace(tzinfo=timezone.utc).timestamp(),
        # 電話掛斷時間，UTC timestamp
        'hang_up_time': hang_up_datetime.replace(tzinfo=timezone.utc).timestamp(),
        'call_duration': call_duration  # 電話通話時間(秒)，記錄通話時長
    }

    payload = {
        'client_id': 'test_client_id',  # 顧客 Hash ID，顧客加密後ID
        'business_unit': 'IF_C170',  # 業管單位，例如客服中心: contact_center
        'request_id': "ckn56cRs8f1KS12345",  # 請求識別碼，唯一碼，用以識別不同次呼叫
        "api_version": "v1.0.0",
        'inputs': json.dumps(inputs)
    }
    payload['business_unit'] = business_unit
    if env_now == 'Aicloud':
        pass
    elif env_now == 'Uat':
        payload['client_id'] = "z+QKW7vrZ+qAJGWA35zpHHOtu0gjzl9reP6LXH+UEQnl46y0xvuQChXGzf+mBEDP"
    elif env_now in ['Prod', 'Staging']:
        payload['client_id'] = "z+QKW7vrZ+qAJGWA35zpHOne3MOdbaft6PW/ErQGrx6K3IYwJtmWMUOi1NUTAdID"
    return payload


def generate_files(file_name='00003.wav'):
    py_dir = "/".join(__file__.split('/')[:-1])
    files = {
        'file': open(f'{py_dir}/test_wavs/{file_name}', 'rb')
    }
    return files


def call_api(
        env_now,
        project_name,
        file_name='00003.wav',
        business_unit='IF_C170'):
    if env_now != 'Aicloud':
        full_hostname = APITestInfo.get_hostname_info(env_now)
    else:
        full_hostname = 'http://localhost:5000'
    try:
        res = requests.post(
            url=f'{full_hostname}/{project_name}/stt',
            data=construct_api_payload(
                env_now=env_now,
                business_unit=business_unit),
            files=generate_files(
                file_name=file_name))
        print(res.status_code)
        print(res.text)
    except Exception as e:
        print(e)
        sys.exit(1)
    if res.status_code != 200:
        sys.exit(1)


try:
    env_now = APITestInfo.get_env()
except BaseException:
    env_now = 'Aicloud'

for business_unit in ['IF_C170', 'C308', 'CC-C251']:
    for file_name in ['00001.wav', '00002.wav', '00003.wav', '00004.wav']:
        call_api(
            env_now=env_now,
            project_name='if_stt',
            file_name=file_name,
            business_unit=business_unit)


'''
- [o] client_id: str  # 顧客 Hash ID，顧客加密後ID
- [o] business_unit: str = 'contact_center'  # 業管單位，例如客服中心: contact_center
- [o] request_id: str  # 請求識別碼，唯一碼，用以識別不同次呼叫
- [V] client_number: str  # 顧客電話號碼，顧客撥入電話碼，若匿名請留空
- [V] agent_id: str  # 客服員編，用以識別接聽人員
- [V] agent_number: str  # 客服分機號碼，用以識別接聽座席
- [V] dial_in_time: datetime.datetime  # 電話撥入時間，UTC timestamp
- [V] hang_up_time: datetime.datetime  # 電話掛斷時間，UTC timestamp
- [V] call_duration: int  # 電話通話時間(秒)，記錄通話時長
'''
