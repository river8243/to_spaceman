#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import os
from datetime import timezone
import datetime


class UtcTime:
    def __init__(self):
        pass

    @staticmethod
    def get_current_utc_time():
        dt = datetime.datetime.now(timezone.utc)
        utc_time = dt.replace(tzinfo=timezone.utc)
        utc_timestamp = utc_time.timestamp()
        return utc_timestamp


class FileManager:

    def __init__(self, full_path, source=None, target=None):
        pass

    @staticmethod
    def list_all_file_type_in_dir(
            regex='\\.wav$|\\.mp3$|\\.aac$|\\.flac$|\\.ape$|\\.ogg$|\\.m4a$|\\.wma$|\\.vox$',
            path=None):
        rule = re.compile(regex)
        if path is None:
            print('沒輸入路徑')
            return None
        elif not isinstance(path, str):
            print('path變數不是字串')
            return None
        elif path[0] != '/':
            print('請輸入絕對路徑')
            return None
        file_list = list()
        for dirpath, dirnames, files in os.walk(path, topdown=True):
            for dirs in dirnames:
                if re.search('ipynb', dirs) is not None:
                    dirnames.remove(dirs)
            for name in files:
                if rule.search(name) is not None:
                    path = os.path.join(dirpath, name)
                    file_list.append(path)
        # file_list.sort()
        return file_list
