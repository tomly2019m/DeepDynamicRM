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

DAY_NIGHT_CONFIG = {"period": 250, "base_iat": 1, "peak_iat": 4, "day_start": 0.2, "day_duration": 0.6}


class SocialMediaUser(HttpUser):
    _cycle_start = time.time()  # 记录测试开始时间

    def wait_time(self):
        elapsed = time.time() - self._cycle_start
        cycle_pos = elapsed % DAY_NIGHT_CONFIG["period"]
        transition = 30  # 固定30秒过渡区间

        day_start = DAY_NIGHT_CONFIG["period"] * DAY_NIGHT_CONFIG["day_start"]
        day_end = day_start + DAY_NIGHT_CONFIG["period"] * DAY_NIGHT_CONFIG["day_duration"]

        # 计算白天核心区间
        core_day_start = day_start + transition
        core_day_end = day_end - transition

        # 计算过渡因子
        if cycle_pos < day_start - transition:
            factor = 0.0
        elif cycle_pos < day_start + transition:
            # 进入白天的斜坡上升
            factor = (cycle_pos - (day_start - transition)) / (2 * transition)
        elif cycle_pos < core_day_end:
            factor = 1.0
        elif cycle_pos < day_end + transition:
            # 离开白天的斜坡下降
            factor = 1 - (cycle_pos - core_day_end) / (2 * transition)
        else:
            factor = 0.0

        # 限制因子范围
        factor = max(0.0, min(1.0, factor))

        current_iat = DAY_NIGHT_CONFIG["base_iat"] + \
                     (DAY_NIGHT_CONFIG["peak_iat"] - DAY_NIGHT_CONFIG["base_iat"]) * (1 - factor)

        return max(0.05, np.random.exponential(current_iat))

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
