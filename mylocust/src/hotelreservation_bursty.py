import random
from locust import HttpUser, task, tag, between
import base64
import os
from pathlib import Path
import logging
import numpy as np
import time
import json

import locust.stats

locust.stats.CSV_STATS_INTERVAL_SEC = 0.5  # second

random.seed(time.time())

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# logging.basicConfig(level=logging.INFO,
#                     # filename='/mnt/locust_log/locust_openwhisk_log.txt',
#                     # filemode='w+',
#                     format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.basicConfig(level=logging.INFO)


def get_user():
    user_id = random.randint(0, 500)
    user_name = 'Cornell_' + str(user_id)
    password = ""
    for i in range(0, 10):
        password = password + str(user_id)
    return user_name, password


mean_iat = 1

# 突发负载配置
BURST_CONFIG = {"base_iat": 1, "burst_iat": 0.7, "cycle_duration": 400, "burst_duration": 100, "min_interval": 150}


class SocialMediaUser(HttpUser):
    _test_start = time.time()  # 测试启动时间
    _burst_schedule = {}  # 存储周期对应的突发时间

    def get_current_cycle(self):
        """计算当前所处的周期编号和周期内时间"""
        elapsed = time.time() - self._test_start
        cycle_num = int(elapsed // BURST_CONFIG["cycle_duration"])
        cycle_pos = elapsed % BURST_CONFIG["cycle_duration"]
        return cycle_num, cycle_pos

    def get_burst_window(self, cycle_num):
        """获取或创建当前周期的突发时间窗口"""
        if cycle_num not in self._burst_schedule:
            # 生成随机突发起始时间（确保不重叠）
            prev_end = self._burst_schedule.get(cycle_num - 1, {}).get('end', 0)
            min_start = max(0, prev_end + BURST_CONFIG["min_interval"] - BURST_CONFIG["cycle_duration"])

            start = random.uniform(min_start, BURST_CONFIG["cycle_duration"] - BURST_CONFIG["burst_duration"])

            self._burst_schedule[cycle_num] = {'start': start, 'end': start + BURST_CONFIG["burst_duration"]}
        return self._burst_schedule[cycle_num]

    def wait_time(self):
        cycle_num, cycle_pos = self.get_current_cycle()
        burst = self.get_burst_window(cycle_num)

        # 判断是否在突发时段
        if burst['start'] <= cycle_pos < burst['end']:
            # 突发期间使用高强度负载
            return max(0.05, np.random.exponential(BURST_CONFIG["burst_iat"]))

        # 非突发期间使用基础负载
        return max(0.05, np.random.exponential(BURST_CONFIG["base_iat"]))

    @task(600)
    @tag('search_hotel')
    def search_hotel(self):
        in_date = random.randint(9, 23)
        out_date = random.randint(in_date + 1, 24)
        if in_date <= 9:
            in_date = "2015-04-0" + str(in_date)
        else:
            in_date = "2015-04-" + str(in_date)

        if out_date <= 9:
            out_date = "2015-04-0" + str(out_date)
        else:
            out_date = "2015-04-" + str(out_date)

        lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0

        url = '/hotels?inDate=' + in_date + '&outDate=' + out_date + \
            '&lat=' + str(lat) + "&lon=" + str(lon)

        r = self.client.get(url, name='search_hotel', timeout=10)
        if r.status_code > 202:
            logging.warning('search_hotel resp.status = %d, text=%s' % (r.status_code, r.text))

    @task(390)
    @tag('recommend')
    def recommend(self):
        coin = random.random()
        if coin < 0.33:
            req = 'dis'
        elif coin < 0.66:
            req = 'rate'
        else:
            req = 'price'

        lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0

        url = '/recommendations?require=' + req + \
            "&lat=" + str(lat) + "&lon=" + str(lon)

        r = self.client.get(url, name='recommend', timeout=10)
        if r.status_code > 202:
            logging.warning('recommend resp.status = %d, text=%s' % (r.status_code, r.text))

    @task(5)
    @tag('reserve')
    def reserve(self):
        in_date = random.randint(9, 23)
        out_date = random.randint(in_date + 1, 24)

        if in_date <= 9:
            in_date = "2015-04-0" + str(in_date)
        else:
            in_date = "2015-04-" + str(in_date)

        if out_date <= 9:
            out_date = "2015-04-0" + str(out_date)
        else:
            out_date = "2015-04-" + str(out_date)

        lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0

        hotel_id = str(random.randint(1, 80))
        user_name, password = get_user()

        num_room = 1

        url = '/reservation?inDate=' + in_date + "&outDate=" + out_date + \
            "&lat=" + str(lat) + "&lon=" + str(lon) + "&hotelId=" + hotel_id + \
            "&customerName=" + user_name + "&username=" + user_name + \
            "&password=" + password + "&number=" + str(num_room)

        r = self.client.post(url, name='reserve', timeout=10)

    @task(5)
    @tag('user_login')
    def read_user_timeline(self):
        user_name, password = get_user()
        url = '/user?username=' + user_name + "&password=" + password

        r = self.client.get(url, name='user_login', timeout=10)

        if r.status_code > 202:
            logging.warning('read_user_timeline resp.status = %d, text=%s' % (r.status_code, r.text))
