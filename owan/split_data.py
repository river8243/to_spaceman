from random import shuffle, sample
from collections import defaultdict
from tqdm import tqdm
import pickle
import time

### 733random
# groups_1 = list(range(0, 4))
# groups_2 = list(range(4, 8))
# groups_3 = list(range(8, 408))
# groups_4 = list(range(408, 733))

# train, valid, test = [], [], []
# shuffle(groups_1), shuffle(groups_2), shuffle(groups_3), shuffle(groups_4)
# train = groups_1[:2] + groups_2[:2] + groups_3[:int(400*0.7)] + groups_4[:int(325*0.7)]
# valid = [groups_1[2]] + [groups_2[2]] + groups_3[int(400*0.7):int(400*0.8)] + groups_4[int(325*0.7):int(325*0.8)]
# test = [groups_1[3]] + [groups_2[3]] + groups_3[int(400*0.8):] + groups_4[int(325*0.8):]



with open('/home/jovyan/wm-insur-call-qa/owen/clean_data/real_call_raw.pkl', 'rb') as f:
    wm_dict = pickle.load(f)

wm_dict_raw = defaultdict(list)
num = 0
for k, v in wm_dict.items():
    if len(v) < 100:
        continue
    wm_dict_raw[num] = v
    num += 1

groups_1 = list(range(0, num))
train, valid, test = groups_1[:57], groups_1[57:65], groups_1[65:]
# y = [y_head.tolist().index(1) for y_head in y]

# speaker_dict = defaultdict(list)
# for X_head, y_head in tqdm(zip(X, y)):
#     speaker_dict[y_head] += [X_head.tolist()]

train_dict, valid_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list)
for i in tqdm(train):
    train_dict[i] = wm_dict_raw[i]
for i in tqdm(valid):
    valid_dict[i] = wm_dict_raw[i]
for i in tqdm(test):
    test_dict[i] = wm_dict_raw[i]

with open('tdnn_callqa_train.pkl', 'wb') as fw:
    pickle.dump(train_dict, fw)
with open('tdnn_callqa_valid.pkl', 'wb') as fw:
    pickle.dump(valid_dict,  fw)
with open('tdnn_callqa_test.pkl', 'wb') as fw:
    pickle.dump(test_dict, fw)