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


def f(x):
    """最终版突发函数，严格满足积分条件"""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # 主峰区参数
    peak_center = 1800
    peak_max = 4037
    a = 0.02523  # 精确抛物线系数

    # 非主峰区参数（积分校准）
    base = 1340
    amp1, amp2 = 180, 140

    # 第一段: 0-1440秒
    mask1 = (x >= 0) & (x <= 1440)
    wave1 = base + amp1 * np.sin(4 * np.pi * x[mask1] / 1440)
    wave1 += amp2 * np.sin(8 * np.pi * x[mask1] / 1440)
    result[mask1] = np.maximum(768, wave1)

    # 主峰区: 1440-2160秒
    mask2 = (x > 1440) & (x <= 2160)
    result[mask2] = peak_max - a * (x[mask2] - peak_center)**2

    # 第三段: 2160-3600秒
    mask3 = (x > 2160) & (x <= 3600)
    wave3 = base + amp1 * np.sin(4 * np.pi * (x[mask3] - 2160) / 1440)
    wave3 += amp2 * np.sin(8 * np.pi * (x[mask3] - 2160) / 1440)
    result[mask3] = np.maximum(768, wave3)

    return result


class SocialMediaUser(HttpUser):
    # 添加实例变量记录启动时间
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = self.environment.runner.stats.start_time if self.environment.runner else time.time()

    # 修改后的wait_time方法
    def wait_time(self):
        # 计算已运行时间（秒）
        elapsed = time.time() - self.start_time
        wait = 3000 / f(elapsed)
        return wait

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
