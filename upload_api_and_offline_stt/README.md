# 前置工作: 
1. 至exchanging_pool/to奕勳/stt_model_to_s3

2. 執行pip install . --user 以安裝語音轉文字套件 (SttModel package)

3. 參考stt_model_to_s3/Readme.md 方式確認語音轉文字套件運作正確

# 測試方式
1. 啟 api server
python3 app_laborcall_v100.py

2. request post (先於這支程式修改夾檔名稱)
python3 apitests/stt/request_test.py

# 取得 file 後邏輯實作
在 api_service_*.py 的 run function 中，可以透過 self.file 取得夾檔

# 相關路徑都在以下 config，請在 python 中用 from configobj import ConfigObj
offline_stt_api_config.ini

# API 格式 (暫定)
## 上傳 API 的呼叫者需要傳過來的欄位訊息
- client_id: str  # 顧客 Hash ID，顧客加密後ID
- business_unit: str = 'contact_center'  # 業管單位，例如客服中心: contact_center
- request_id: str  # 請求識別碼，唯一碼，用以識別不同次呼叫
- client_number: str  # 顧客電話號碼，顧客撥入電話碼，若匿名請留空
- agent_id: str  # 客服員編，用以識別接聽人員
- agent_number: str  # 客服分機號碼，用以識別接聽座席
- dial_in_time: datetime.datetime  # 電話撥入時間，UTC timestamp
- hang_up_time: datetime.datetime  # 電話掛斷時間，UTC timestamp
- call_duration: int  # 電話通話時間(秒)，記錄通話時長

## 上傳 api 收到 request 後要回覆給 client 的訊息
- request_id: str  # 請求識別碼，就跟丟過來的一樣	
- trace_id: str  # 追蹤識別碼 	
- request_time: datetime.datetime  # 請求時間，UTC timestamp
- response_time: datetime.datetime  # 回應時間，UTC timestamp
- duration_time: datetime.datetime  # 回應時間 - 請求時間	
- response_code: str  # 回覆狀態碼
    所有可能的選項:
    "201": 正常
    "202": Warning 訊息: 欄位異常 (e.g., 要求A, B欄位，給了A, B, C欄位，多了欄位C)
    "401": 欄位值不符合規範
    "405": 辨識系統忙碌中，請稍候重打 
    "499": 未預期錯誤
- response_message: dict  # 回覆狀態碼內容
    所有可能的選項:
    "201": {"message": "success"}
    "202": {"warning_inputs": ["cust_no", "date"]}
    "401": {"error_inputs": ["business_unit", " dial_in_time "]}
    "405": {"error_message": "system busy, try it later."}
    "499": {"error_message": "Unexpected error occurred."}


## 上傳 API 收到 request 後要記錄在 ES 的訊息 (存成 .json 或 .log，一個 wav 對一個 json，主檔名兩邊要一樣)
- 上述兩大項的欄位，去除重複的
- upload_api_version: str = '100'  # 代表上傳 api 的版本
- upload_time = datetime.datetime  # 上傳此檔案時間，UTC timestamp

(沒辦法取得request_time, response_time, duration_time, 因為是在mlaas底層) 
(沒辦法取得client_id，可能是子悠的 laborcall_framework_v100.py 未把此參數加到object variable裡面，已經修改加入) 


## stt_service 辨識完成時要記錄以下訊息成 json
- 上述所有欄位
- stt_service_version: str  # 代表 stt service 的版本
- recognizer_version: str  # 代表 recognizer.py 的版本
- vosk_version: str  # 代表 vosk 套件使用的版本
- sample_rate: int  # 聲音的採樣頻率 16000 或 8000
- bit_depth: int  # 位元深度，通常是 16
- num_of_sample: int  # 音檔樣本總數
- duration: float  # 從音檔這邊看到的時長
- recog_start_time: datetime.datetime  # 開始進行辨識的時間，UTC timestamp
- recog_end_time: datetime.datetime  # 辨識結束的時間，UTC timestamp
- recog_time: datetime.datetime  # 上兩個欄位值相減
- time_zone: str = str(timezone.utc)  # 時區
- am_version: str = '1.0'   # 聲學模型版本 (可能會是英數字組合)
- lm_version: str = '1.1'   # 語言模型版本 (可能會是英數字組合)
- hostname: str   # 哪一台機器做辨識的，辨識是 mlaas 的哪一台機器


## stt_service life cycle
- 流程：
  1. mlaas 把 api 起來時，在 init 裡面把 ```stt_service.py``` 的 ```stt_scheduler()``` 啟起來
  2. scheduler 每隔一段時間去檢查是否有新的音檔
    1. 若有音檔就丟辨識程序
      1. 目前是走 ```stt_service.stt_inference_parallel()```
  3. mlaas 若關掉 api server，則 ```stt_service.scheduler 理當也要被關掉```
- behaviour of ```stt_service.stt_inference_parallel```
  1. 以 ```multiprocessing``` 處理多工
  2. 根據 config 裡面定的 job 數決定 pool 的 process 數，並開 pool
  3. 把音檔 list 丟進 pool，pool 裡面的 processes 會輪流執行 ```stt_service.single_audio_stt_inference()```


