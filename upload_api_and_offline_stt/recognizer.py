import io
import json
from typing import List

# import fleep
# from fastapi import File, UploadFile
from pydub import AudioSegment
from vosk import KaldiRecognizer, Model
# import aiofiles
import time
import os
import struct


class Recognizer:
    version = '0.1.0'

    def __init__(self, model: Model):
        self._rec = KaldiRecognizer(model, 16000)
        self._type_mapping = {
            "x-wav": "wav",
            "mpeg": "mp3",
            "wav": "wav",
            "wave": "wav"}

    def __del__(self):
        self._rec = None

    def recognize(self, contents: bytes) -> str:

        if self._rec.AcceptWaveform(contents):
            pass

        return json.loads(self._rec.Result())

    def recognize_wav_from_path(self, wav_path: str) -> str:
        # wav_path = '/home/hans/Downloads/chao-16k-16bit.wav'
        while not os.path.exists(wav_path):
            time.sleep(1)
        # async with aiofiles.open(wav_path, 'rb') as f:
        #     wav = await f.read()
        with open(wav_path, 'rb') as f:
            wav = f.read()
            sample_rate = struct.unpack('i', wav[24:28])[0]
            bit_depth = struct.unpack('i', wav[34:36] + b'\x00\x00')[0]
            num_of_sample = int(struct.unpack(
                'i', wav[40:44])[0] / (bit_depth / 8))
            duration = num_of_sample / float(sample_rate)
            print(
                sample_rate,
                "====",
                bit_depth,
                '====',
                num_of_sample,
                '====',
                duration)
        if self._rec.AcceptWaveform(wav):
            pass
        # log_name = wav_path.split('/', -1)[-1].rsplit('.', 1)[0]
        recog_res = json.loads(self._rec.Result())
        recog_res['sample_rate'] = sample_rate
        recog_res['bit_depth'] = bit_depth
        recog_res['duration'] = duration
        recog_res['num_of_sample'] = num_of_sample

        return recog_res

    # def format_normalize(self, file: File, type: str = "wav") -> bytes:
    #     audio = AudioSegment.from_file(file, self._type_mapping[type])
    #     audio = audio.set_frame_rate(16000)

    #     buf = io.BytesIO()
    #     audio.export(buf, format="wav")

    #     return buf.getvalue()
