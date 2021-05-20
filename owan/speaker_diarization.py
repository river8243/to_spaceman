import pickle
import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm
from python_speech_features import mfcc, logfbank
from sklearn.cluster import KMeans

class Speaker_Diarization():
    """
    輸出speaker diarization的秒數，目前僅支援2位語者。

    Input：
    TDNN_model_file:模型路徑
    wav_file:音檔路徑
    ms:音檔切割方式為幾毫秒，目前僅支援2、5、10、15
    sl_ms:音檔切割在進行sliding window時，以幾毫秒做為單位
    parts:sliding window完要進行投票時，要切幾段，並取中間那幾段進行加權
    threshold:sliding window完要進行投票時，加權的倍數

    Output：
    sec:語者交換的秒數
    """
    def __init__(self, TDNN_model_file, wav_file, ms, sl_ms=1, parts=5, weight=2, threshold=0.25):
        self.check_ms(ms)
        self.check_wav(wav_file)
        self.check_model(TDNN_model_file)
        self.ms = ms
        self.sl_ms = sl_ms
        self.parts = parts
        self.weight = weight
        self.threshold = threshold
        self.max_total_context_test, self.wav_long = self.get_wav_para(ms)
        self.rate, self.wav = wavfile.read(wav_file)
        self.net_xvec = torch.load(TDNN_model_file)[:-1]

    @staticmethod
    def check_ms(ms):
        """
        確認切割的毫秒數為目前可支援的毫秒數，僅可支援[2, 5, 10, 15]毫秒
        """
        assert ms in [2, 5, 10, 15], 'Please let ms in [2, 5, 10, 15].'

    @staticmethod
    def check_wav(wav):
        """
        確認音檔有辦法讀取
        """
        assert wavfile.read(wav)

    @staticmethod
    def check_model(model):
        """
        確認TDNN pre train model有辦法讀取
        """
        assert torch.load(model)[:-1]

    def get_wav_para(self, ms):
        """
        取得mfcc函數使用的參數
        """
        para_dict = {15: {'max_total_context_test': 149,
                          'wav_long': 12000},
                     10: {'max_total_context_test': 99,
                          'wav_long': 8000},
                     5: {'max_total_context_test': 49,
                         'wav_long': 4000},
                     2: {'max_total_context_test': 19,
                         'wav_long': 1600}}
        max_total_context_test = para_dict[ms]['max_total_context_test']
        wav_long = para_dict[ms]['wav_long']
        return max_total_context_test, wav_long

    def chunks_sl(self, l, n, k):
        """
        Yield successive n-sized chunks from l by sliding window
        """
        for i in range(0, len(l), int(n/k)):
            yield l[i:i + n]

    def chunks(self, l, n):
        """
        Yield successive n-sized chunks from l
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def wav_expansion(self, rate_, wav_, wav_long_, ms_, sl_ms_, max_total_context_test_):
        """
        對音檔利用mfcc函數，展開成TDNN模型input層的架構
        """
        _min,_max = float('inf'),-float('inf')
        X_test = []
        for chunked_wav in tqdm(self.chunks_sl(wav_, wav_long_, (ms_/sl_ms_))):
            X_sample = mfcc(chunked_wav, samplerate=rate_, numcep=24, nfilt=26, nfft=1024)
            _min = min(np.amin(X_sample),_min)
            _max = max(np.amax(X_sample),_max)
            for chunked_X_sample in list(self.chunks(X_sample, max_total_context_test_)):
                
                    if len(chunked_X_sample) == max_total_context_test_:
                        X_test.append(chunked_X_sample)
        X_test_f = (X_test - _min) / (_max-_min)
        return X_test_f

    def get_xvector(self, X_test_f_, net_xvec_):
        """
        利用TDNN模型取得x-vector
        """
        ng_outout = net_xvec_(torch.tensor(X_test_f_[0:500]).cuda())
        np_ng_output = ng_outout.cpu().detach().numpy()
        np_ng_output_all = np_ng_output
        if len(X_test_f_)>500:
            for i in tqdm(range(1, int(len(X_test_f_)/500)+1)):
                ng_outout = net_xvec_(torch.tensor(X_test_f_[i*500: (i+1)*500]).cuda())
                np_ng_output = ng_outout.cpu().detach().numpy()
                np_ng_output_all = np.append(np_ng_output_all, np_ng_output, axis=0)
        return np_ng_output_all

    def get_voting_prob(self, data, ms_, sl_ms_, parts_, weight_, threshold_):
        """
        對KMeans分群完的結果進行加權投票，區分為第一個語者或第二個語者
        """
        avg_list=[]
        result_list = []
        for t in tqdm(range(1, len(data)+1)):
            p_list = data[max(0, (t-int(ms_/sl_ms_))): t]
            w_num = int(len(p_list)/parts_)
            if w_num > 0:
                first_list = p_list[0: w_num]
                medium_list = p_list[w_num: -w_num]
                last_list = p_list[-w_num:]
            else:
                first_list = []
                medium_list = p_list
                last_list = []
            w_avg = (sum(first_list) + weight_ * sum(medium_list) + sum(last_list)) / (len(first_list) + weight_ * len(medium_list) + len(last_list))
            avg_list.append(w_avg)
            if sum(data) >= len(data)/2:
                if w_avg >= threshold_:
                    result_list.append(1)
                else:
                    result_list.append(0)
            else:
                if w_avg >= (1-threshold_):
                    result_list.append(1)
                else:
                    result_list.append(0)
        return avg_list, result_list

    def get_sec(self, data, sl_ms):
        """
        整理語者交換的秒數
        """
        sec = [0]
        peo = [data[0]]
        for i in tqdm(range(1, len(data))):
            if data[i] != data[i-1]:
                sec.append(round(i*sl_ms*0.1, 3))
                peo.append(data[i])
        return sec, peo

    def speaker_diarization_pipeline(self):
        """
        語者分離的pipeline
        """
        # 展開音檔
        X_test_f = self.wav_expansion(self.rate, self.wav, self.wav_long, self.ms, self.sl_ms, self.max_total_context_test)
        # 得到x-vector
        x_vector = self.get_xvector(X_test_f, self.net_xvec)
        # 分群
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x_vector)
        kmeans_result = kmeans.labels_
        # 加權投票
        avg_list, result_list = self.get_voting_prob(kmeans_result, self.ms, self.sl_ms, self.parts, self.weight, self.threshold)
        # 整理語者交換的秒數
        sec, peo = self.get_sec(result_list, self.sl_ms)
        return sec



