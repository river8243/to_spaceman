'''
## TextGrid 
[tgt git](https://github.com/hbuschme/TextGridTools)
[tgt doc](https://textgridtools.readthedocs.io/en/stable/api.html)

## Praat
[script doc](https://www.fon.hum.uva.nl/praat/manual/Scripting.html)
[小狐狸簡介](http://yhhuang1966.blogspot.com/2019/12/praat_9.html)
[parselmouth github](https://github.com/YannickJadoul/Parselmouth)
[parselmouth doc](https://parselmouth.readthedocs.io/en/stable/)

written 2020/12/25 by yihsuan
'''

import os
import tgt
from parselmouth import praat
from parselmouth import TextGrid

class PraatTextgrid(object):
    def __init__(self, name):
        self.name = name

    @staticmethod
    def get_tiernames_from_tgfile(read_file):
        tg = tgt.read_textgrid(read_file, encoding='utf-16')
        tier_names = [tier.name for tier in tg.tiers]
        return tg, tier_names
        
    @staticmethod
    def read_tier_from_tg(tg, tier_name):
        tag_tier = tg.get_tier_by_name(tier_name)
        return tag_tier

    @staticmethod
    def write_tgfile(outfile, tiers):
        tg = tgt.core.TextGrid()
        for tier in tiers:
            tg.add_tier(tier)
        tgt.io.write_to_file(tg, outfile, format = 'long', encoding='utf-8')

    @staticmethod
    def create_interval_tier(start_time, end_time, name, intervals):
        tier = tgt.core.IntervalTier(start_time, end_time, name)
        for interval in intervals:
            # interval = [start_time, end_time, text]
            annotation = tgt.core.Interval(interval[0], interval[1], interval[2])
            tier.add_interval(annotation)
        return tier

    @staticmethod
    def create_point_tier(start_time, end_time, name, points):
        tier = tgt.core.IntervalTier(start_time, end_time, name)
        for interval in intervals:
            # interval = [time, text]
            annotation = tgt.core.Interval(interval[0], interval[1])
            tier.add_interval(annotation)
        return tier
        
    @staticmethod
    def create_silence_tg(wav_path, output_sil_tg_path):
        wav_praat_read = praat.call('Read from file', wav_path)
        sil_tier = praat.call(wav_praat_read, 'To TextGrid (silences)', 100, 0, -25, 0.1, 0.1, "SIL", "")
        output = praat.call(sil_tier, 'Write to text file', output_sil_tg_path)

    @staticmethod
    def make_silence_tier(wav_path,
                          min_pitch_hz=100.0,
                          time_step_sec=0.0,
                          sil_threshold_db=-25.0,
                          min_sil_dur_sec=0.1,
                          min_sound_dur_sec=0.1,
                          sil_interval_name='SIL',
                          sound_interval_name='') -> tgt.core.TextGrid:
        wav_praat_read = praat.call('Read from file', wav_path)
        sil_tier = praat.call(wav_praat_read, 'To TextGrid (silences)', min_pitch_hz, time_step_sec, sil_threshold_db, min_sil_dur_sec, min_sound_dur_sec, sil_interval_name, sound_interval_name)
        sil_tgt = TextGrid.to_tgt(sil_tier)
        return sil_tgt
        
        
def get_wave_duration(wav_file):
        import sox
        duration = sox.file_info.duration(wav_file)
        return duration

if __name__=='__main__':
    root = './'
    test_wav = os.path.join('./test.wav')
    test_read_tg = os.path.join(root, 'praat_test_sil.TextGrid')
    test_write_siltier_tg = os.path.join(root, 'test_out_sil.TextGrid')
    test_write_tg = os.path.join(root, 'test_out.TextGrid')
    
    # write silence annotated TextGrid
    PraatTextgrid.create_silence_tg(test_wav, test_write_siltier_tg)  # 寫出 silence annotation TextGrid 檔案
    sil_tg = PraatTextgrid.make_silence_tier(test_wav)  # 回傳 tgt.core.TextGrid 格式
    
    # write self-annotated TextGrid
    wav_dur = get_wave_duration(test_wav)
    intervals = [[0, 1, 'ini'], [2, 3, 'sec'], [6, wav_dur, 'end']] # intervals cannot be overlapped
    int_tier = PraatTextgrid.create_interval_tier(0, wav_dur, 'interval_tier', intervals)
    
    points = [[3, 'ini'], [7, 'sec'], [10, 'end']]
    pt_tier = PraatTextgrid.create_point_tier(0, wav_dur, 'point_tier', points)
    
    PraatTextgrid.write_tgfile(test_write_tg, [int_tier, pt_tier])

    # read TextGrid
    textgrid, tier_names = PraatTextgrid.get_tiernames_from_tgfile(test_read_tg)
    for name in tier_names:
        tier = PraatTextgrid.read_tier_from_tg(textgrid, name)
        print(f"{name}")
        for annotation in tier.annotations:
            print(f"{annotation}") 
            # 可以直接輸出 annotation.start_time, annotation.end_time, annotation.text
            # 如果 annotation.start_time == annotation.end_time 則 tier 是 point tier
