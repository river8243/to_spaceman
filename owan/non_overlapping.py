import numpy as np
import sys

sys.path.append('/home/jovyan/wm-insur-call-qa/owen/evaluation')
from textgrid_open import get_textgrid_dict

def determine_abs(x, y):
    x, y = [i+1 if i == 0 else i for i in [x, y]]
    return np.sign(x * y)

def get_non_overlapping_time(x_time, y_time):
    only_x_time = []
    for x_start, x_end in x_time:
        revised_tag = False
        for y_start, y_end in y_time:
            if x_end <= y_start or y_end <= x_start:
                continue
            revised_tag = True
            y_start_relative = determine_abs(*[y_start - e for e in [x_start, x_end]])
            y_end_relative = determine_abs(*[y_end - e for e in [x_start, x_end]])
            if (y_start_relative, y_end_relative) == (-1, -1):
                only_x_time.append((x_start, y_start))
                only_x_time.append((y_end, x_end))
            elif (y_start_relative, y_end_relative) == (-1, 1):
                only_x_time.append((x_start, y_start))
            elif (y_start_relative, y_end_relative) == (1, -1):
                only_x_time.append((y_end, x_end))
            else:
                # 完全被覆蓋
                pass
        if not revised_tag:
            only_x_time.append((x_start, x_end))
    return only_x_time

if __name__ == "__main__":
    TEXTGRID_PATH = '/home/jovyan/wm-insur-call-qa/insur_data_202012_result/20201203/220R0000000127160697593000042.TextGrid'
    conversation = get_textgrid_dict(TEXTGRID_PATH)
    esun_time = conversation['Esun']['times']
    cust_time = conversation['customer']['times']
    only_esun_time = get_non_overlapping_time(esun_time, cust_time)
    only_cust_time = get_non_overlapping_time(cust_time, esun_time)
    